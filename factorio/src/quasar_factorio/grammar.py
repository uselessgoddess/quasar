"""The structural token grammar the model actually learns.

A blueprint is written as whitespace-separated tokens, one document per
blueprint:

    <bp> <spec> k:smelter-column r:iron-plate #12 #14 #22 </spec>
      <e> stone-furnace x00 y00 d0
      <e> transport-belt x02 y00 d2
      ... </bp>

Three decisions are doing the work here.

*Coordinates are their own tokens*, `x00`-`x63` and `y00`-`y63`, rather than
digits. A digit-pair spelling makes the model learn that `1` `2` next to an `x`
means twelve, and it makes off-by-ten errors cheap; one token per tile makes
position a lookup and puts the whole grid in the embedding table at a cost of
128 rows.

*Entities have no closing tag.* Whether a recipe token follows is decided by the
prototype table, so arity is known from the name and `</e>` would be four
thousand wasted tokens per training batch.

*The spec comes first and is dropped at random.* One model then serves both
unconditional sampling and "build me twelve furnaces smelting iron", without a
second finetuning stage to go wrong.

The spec grew two optional sections for the module task — a zone with declared
inputs and outputs, and the recipe chain that fills it:

    <bp> <spec> k:module r:electronic-circuit #5 #16 #20
      <in> i:copper-plate s:n x03 y00
      <in> i:iron-plate s:n x00 y06
      <out> i:electronic-circuit s:e x15 y12
      <plan> assembling-machine-1 r:copper-cable #2
             assembling-machine-1 r:electronic-circuit #3 </plan>
      </spec> <e> ... </bp>

Both are optional and both are absent from every document the older generators
write, so the grammar is a superset of the one that came before it: a corpus
mixing modules and belt lanes needs one vocabulary and one model.

A port is written as the tile it feeds *inside* the design plus the edge it sits
on, rather than as an offset along that edge. Two reasons: the tile transforms
under rotation exactly like an entity does, so augmentation cannot desynchronise
it from the design; and the model already knows what `x03 y00` means, whereas an
offset would be a third numbering scheme with the same shape as the other two.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .blueprint import GRID, Blueprint, BlueprintError, Placement
from .prototypes import Data, load

BP, BP_END = "<bp>", "</bp>"
SPEC, SPEC_END = "<spec>", "</spec>"
ENT = "<e>"
IN, OUT = "<in>", "<out>"
PLAN, PLAN_END = "<plan>", "</plan>"
UNK = "<unk>"
EOS = "<|endoftext|>"
NO_RECIPE = "r:none"

# Underground belts carry an entrance/exit flag in the blueprint JSON that is
# independent of their direction, so it gets its own slot in the grammar.
FLOWS = ("t:input", "t:output")

# Which edge of the zone a port sits on, clockwise from north so that a quarter
# turn is an index step.
SIDES = ("n", "e", "s", "w")

STRUCTURAL = (UNK, EOS, BP, BP_END, SPEC, SPEC_END, ENT, IN, OUT, PLAN, PLAN_END)

# What a blueprint is *for*. The generators stamp their own kind; scraped
# designs get `real`, which is a real signal — it tells the model that this
# example was not produced by one of the templates.
KINDS = (
    "real",
    "module",
    "belt-lane",
    "smelter-column",
    "assembler-row",
    "mall-cell",
    "mining-outpost",
    "solar-block",
    "lab-block",
    "balancer",
    "oil-block",
    "bus-tap",
    "misc",
)

# Counts and extents share one number vocabulary; they never appear in a
# position where they could be confused with a coordinate.
COUNT_MAX = GRID


def kind_token(kind: str) -> str:
    return f"k:{kind}"


def recipe_token(recipe: str | None) -> str:
    return NO_RECIPE if not recipe else f"r:{recipe}"


def count_token(value: int) -> str:
    return f"#{min(max(value, 0), COUNT_MAX)}"


def x_token(value: int) -> str:
    return f"x{value:02d}"


def y_token(value: int) -> str:
    return f"y{value:02d}"


def direction_token(value: int) -> str:
    return f"d{value % 8}"


def item_token(name: str) -> str:
    return f"i:{name}"


def side_token(side: str) -> str:
    return f"s:{side}"


@dataclass(frozen=True)
class Port:
    """An item crossing the boundary of the zone, at a named tile.

    `x`, `y` is the tile *inside* the design that the item arrives on or leaves
    from, and `side` is the edge of the zone it crosses. Together they are what
    makes a module composable: two modules whose ports agree can be placed next
    to each other and the belts line up, which is not something a design with
    its inputs somewhere in the middle can promise.
    """

    role: str  # "in" or "out"
    item: str
    x: int
    y: int
    side: str = "w"

    def tokens(self) -> list[str]:
        return [
            IN if self.role == "in" else OUT,
            item_token(self.item),
            side_token(self.side),
            x_token(self.x),
            y_token(self.y),
        ]

    def moved(self, dx: int, dy: int) -> Port:
        return Port(self.role, self.item, self.x + dx, self.y + dy, self.side)

    def rotated(self, quarters: int) -> Port:
        """The same tile and edge after `quarters` clockwise turns.

        The arithmetic is `blueprint.rotate`'s, for a 1x1 entity: the caller
        shifts the result to the origin with the same offset the blueprint got,
        which is why `rotate_raw` exists.
        """
        x, y = self.x, self.y
        for _ in range(quarters % 4):
            x, y = -(y + 1), x
        return Port(self.role, self.item, x, y, SIDES[(SIDES.index(self.side) + quarters) % 4])

    def mirrored(self) -> Port:
        side = {"e": "w", "w": "e"}.get(self.side, self.side)
        return Port(self.role, self.item, -(self.x + 1), self.y, side)


@dataclass(frozen=True)
class Step:
    """One line of the plan: this many of this machine, running this recipe."""

    machine: str
    recipe: str
    count: int

    def tokens(self) -> list[str]:
        return [self.machine, recipe_token(self.recipe), count_token(self.count)]


@dataclass(frozen=True)
class Spec:
    """The conditioning prefix: what to build, of what, how much, how big.

    `ports` and `plan` are empty for everything except the module task, and a
    spec with neither serialises to exactly the six tokens it always did.
    """

    kind: str = "misc"
    product: str | None = None
    count: int = 0
    width: int = 0
    height: int = 0
    ports: tuple[Port, ...] = ()
    plan: tuple[Step, ...] = ()

    def tokens(self) -> list[str]:
        tokens = [
            SPEC,
            kind_token(self.kind),
            recipe_token(self.product),
            count_token(self.count),
            count_token(self.width),
            count_token(self.height),
        ]
        for port in self.ports:
            tokens += port.tokens()
        if self.plan:
            tokens.append(PLAN)
            for step in self.plan:
                tokens += step.tokens()
            tokens.append(PLAN_END)
        tokens.append(SPEC_END)
        return tokens

    def inputs(self) -> tuple[Port, ...]:
        return tuple(port for port in self.ports if port.role == "in")

    def outputs(self) -> tuple[Port, ...]:
        return tuple(port for port in self.ports if port.role == "out")

    def to_dict(self) -> dict:
        """A JSON-safe spec, for `prompts.jsonl`."""
        return {
            "kind": self.kind,
            "product": self.product,
            "count": self.count,
            "width": self.width,
            "height": self.height,
            "ports": [asdict(port) for port in self.ports],
            "plan": [asdict(step) for step in self.plan],
        }

    @staticmethod
    def from_dict(payload: dict) -> Spec:
        """The inverse of :meth:`to_dict`.

        Written out rather than left to `Spec(**payload)` because the nested
        rows come back as plain dicts, and a `Spec` holding dicts where it
        promises `Port`s produces a prompt that is missing its ports entirely —
        an eval set that quietly asks a different question than the corpus
        answered. The round trip is asserted in `tests/test_dataset.py`.
        """
        payload = dict(payload)
        ports = tuple(Port(**port) for port in payload.pop("ports", ()) or ())
        plan = tuple(Step(**step) for step in payload.pop("plan", ()) or ())
        return Spec(**payload, ports=ports, plan=plan)

    @staticmethod
    def measure(
        blueprint: Blueprint,
        kind: str,
        product: str | None,
        data: Data,
        *,
        ports: tuple[Port, ...] = (),
        plan: tuple[Step, ...] = (),
        zone: tuple[int, int] | None = None,
    ) -> Spec:
        """Derive a spec from a finished blueprint.

        `count` is the number of *crafting* machines when there are any and the
        total entity count otherwise, because "twelve furnaces" is a meaningful
        request and "twelve belts" is not.

        `zone` overrides the measured extent, which is what a module wants: the
        player marks out an area and the design has to fit *inside* it, so the
        spec states the area asked for rather than the area used.
        """
        width, height = zone if zone is not None else blueprint.extent(data)
        crafters = sum(
            1
            for placement in blueprint.entities
            if (proto := data.entity(placement.name)) is not None and proto.crafting_categories
        )
        return Spec(
            kind=kind,
            product=product,
            count=crafters or len(blueprint.entities),
            width=width,
            height=height,
            ports=ports,
            plan=plan,
        )


def vocabulary(data: Data | None = None) -> list[str]:
    """Every token the model can emit, in a fixed, reproducible order.

    Derived from the prototype table rather than from the corpus: a vocabulary
    that shifts when the dataset is rebuilt invalidates every checkpoint, and
    the unused rows cost 96 floats each.
    """
    data = data or load()
    tokens = list(STRUCTURAL)
    tokens += [kind_token(kind) for kind in KINDS]
    tokens += sorted(data.entities)
    tokens += [x_token(value) for value in range(GRID)]
    tokens += [y_token(value) for value in range(GRID)]
    tokens += [direction_token(value) for value in range(8)]
    tokens += list(FLOWS)
    tokens += [side_token(side) for side in SIDES]
    tokens += [NO_RECIPE] + [f"r:{name}" for name in sorted(data.recipes)]
    # Ports name an *item*, which is not the same set as the recipes: `r:` is a
    # process and `i:` is a thing on a belt. Sharing one set would make
    # `r:copper-cable` mean both "run this recipe" and "carry this item", and
    # the first mismatch — an item with no recipe, a recipe with two results —
    # would be a silent aliasing bug rather than an unknown token.
    tokens += [item_token(name) for name in sorted(data.stack_sizes)]
    tokens += [count_token(value) for value in range(COUNT_MAX + 1)]
    seen, unique = set(), []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def serialise(
    blueprint: Blueprint,
    data: Data | None = None,
    spec: Spec | None = None,
) -> str:
    """Blueprint -> one training document (without the terminator)."""
    data = data or load()
    blueprint = blueprint.normalised(data)
    if not blueprint.fits(data):
        width, height = blueprint.extent(data)
        raise BlueprintError(f"{width}x{height} does not fit the {GRID}x{GRID} grid")

    tokens = [BP]
    if spec is not None:
        tokens += spec.tokens()
    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        if proto is None:
            raise BlueprintError(f"unknown entity {placement.name!r}")
        tokens += [
            ENT,
            placement.name,
            x_token(placement.x),
            y_token(placement.y),
            direction_token(placement.direction),
        ]
        if proto.takes_recipe:
            tokens.append(recipe_token(placement.recipe))
        if proto.takes_flow:
            tokens.append(f"t:{placement.flow or 'input'}")
    tokens.append(BP_END)
    return " ".join(tokens)


def prompt(spec: Spec) -> str:
    """The conditioning half of a document: everything up to the first entity.

    Sampling starts here, so it lives next to `serialise` and is built from the
    same tokens. A prompt assembled by string-slicing a training document would
    drift the first time the spec grows a field.
    """
    return " ".join([BP, *spec.tokens()])


_COORD = re.compile(r"^([xy])(\d{2})$")


class ParseError(ValueError):
    """The token stream did not describe a blueprint.

    Carries how far it got, so evaluation can report *where* generation broke
    rather than only that it did.
    """

    def __init__(self, message: str, *, at: int, parsed: int) -> None:
        super().__init__(message)
        self.at = at
        self.parsed = parsed


def parse(text: str, data: Data | None = None) -> tuple[Blueprint, Spec | None]:
    """One training document -> a blueprint, strictly.

    Strict on purpose. This is the grader: a stream that a real Factorio client
    would refuse must be refused here too, or the validity metric measures
    nothing.
    """
    data = data or load()
    tokens = text.split()
    index = 0

    def error(message: str, parsed: int):
        return ParseError(message, at=index, parsed=parsed)

    while index < len(tokens) and tokens[index] != BP:
        if tokens[index] != EOS:
            raise error(f"expected {BP}, found {tokens[index]!r}", 0)
        index += 1
    if index >= len(tokens):
        raise error(f"no {BP}", 0)
    index += 1

    spec = None
    if index < len(tokens) and tokens[index] == SPEC:
        body = tokens[index + 1 : index + 6]
        if len(body) < 5:
            raise error("malformed spec", 0)
        kind, product, count, width, height = body
        index += 6

        ports: list[Port] = []
        while index < len(tokens) and tokens[index] in (IN, OUT):
            role = "in" if tokens[index] == IN else "out"
            field = tokens[index + 1 : index + 5]
            if len(field) < 4:
                raise error("truncated port", 0)
            item, side, raw_x, raw_y = field
            if not item.startswith("i:") or side not in [side_token(name) for name in SIDES]:
                raise error(f"malformed port {item!r} {side!r}", 0)
            ports.append(
                Port(
                    role=role,
                    item=item.removeprefix("i:"),
                    x=_coordinate(raw_x, "x", error, 0),
                    y=_coordinate(raw_y, "y", error, 0),
                    side=side.removeprefix("s:"),
                )
            )
            index += 5

        plan: list[Step] = []
        if index < len(tokens) and tokens[index] == PLAN:
            index += 1
            while index < len(tokens) and tokens[index] != PLAN_END:
                line = tokens[index : index + 3]
                if len(line) < 3:
                    raise error("truncated plan step", 0)
                machine, recipe, amount = line
                if data.entity(machine) is None:
                    raise error(f"unknown machine {machine!r}", 0)
                if not recipe.startswith("r:") or recipe == NO_RECIPE:
                    raise error(f"expected a recipe, found {recipe!r}", 0)
                plan.append(
                    Step(
                        machine=machine,
                        recipe=recipe.removeprefix("r:"),
                        count=_number(amount, error),
                    )
                )
                index += 3
            if index >= len(tokens):
                raise error(f"no {PLAN_END}", 0)
            index += 1

        if tokens[index : index + 1] != [SPEC_END]:
            raise error("malformed spec", 0)
        spec = Spec(
            kind=kind.removeprefix("k:"),
            product=None if product == NO_RECIPE else product.removeprefix("r:"),
            count=_number(count, error),
            width=_number(width, error),
            height=_number(height, error),
            ports=tuple(ports),
            plan=tuple(plan),
        )
        index += 1

    placements: list[Placement] = []
    while index < len(tokens) and tokens[index] != BP_END:
        if tokens[index] != ENT:
            raise error(f"expected {ENT}, found {tokens[index]!r}", len(placements))
        index += 1
        head = tokens[index : index + 4]
        if len(head) < 4:
            raise error("truncated entity", len(placements))
        name, raw_x, raw_y, raw_direction = head
        proto = data.entity(name)
        if proto is None:
            raise error(f"unknown entity {name!r}", len(placements))
        index += 4

        recipe = None
        if proto.takes_recipe:
            if index >= len(tokens):
                raise error(f"{name} is missing its recipe", len(placements))
            token = tokens[index]
            if token != NO_RECIPE and not token.startswith("r:"):
                raise error(f"{name} is missing its recipe", len(placements))
            recipe = None if token == NO_RECIPE else token.removeprefix("r:")
            if recipe is not None and recipe not in data.recipes:
                raise error(f"unknown recipe {recipe!r}", len(placements))
            index += 1

        flow = None
        if proto.takes_flow:
            if index >= len(tokens) or tokens[index] not in FLOWS:
                raise error(f"{name} is missing its input/output flag", len(placements))
            flow = tokens[index].removeprefix("t:")
            index += 1

        placements.append(
            Placement(
                name=name,
                x=_coordinate(raw_x, "x", error, len(placements)),
                y=_coordinate(raw_y, "y", error, len(placements)),
                direction=_direction(raw_direction, error, len(placements)),
                recipe=recipe,
                flow=flow,
            )
        )

    if index >= len(tokens):
        raise error(f"no {BP_END}", len(placements))
    return Blueprint(entities=placements).normalised(data), spec


def _number(token: str, error) -> int:
    if not token.startswith("#") or not token[1:].isdigit():
        raise error(f"expected a count, found {token!r}", 0)
    return int(token[1:])


def _coordinate(token: str, axis: str, error, parsed: int) -> int:
    match = _COORD.match(token)
    if not match or match.group(1) != axis:
        raise error(f"expected a {axis} coordinate, found {token!r}", parsed)
    return int(match.group(2))


def _direction(token: str, error, parsed: int) -> int:
    if len(token) != 2 or token[0] != "d" or not token[1].isdigit():
        raise error(f"expected a direction, found {token!r}", parsed)
    return int(token[1])
