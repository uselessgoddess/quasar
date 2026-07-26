"""Do items actually get from the inputs to the output?

Every metric in `validate` is local. "Powered" asks whether a pole is nearby,
"connected inserters" asks whether both neighbouring tiles are touchable, "belts
lead somewhere" asks about one tile ahead. A design can score 1.0 on all three
and still be a diorama: a belt of copper plate feeding an assembler set to
`electronic-circuit`, which needs cable, so nothing is ever made. That is exactly
what the measured run found — every local metric at ceiling, no metric able to
see it.

So this module simulates the one thing that matters and nothing else: *which
items can reach which entity*. Not throughput, not stack sizes, not fuel. It is a
reachability fixed point over item sets — seed the input ports, push along belts,
carry across inserters, and let a machine produce its results once its every
ingredient has arrived. Repeat until nothing changes.

Deliberately optimistic. Belt sides, insertion rates and buffering are all
ignored, so a design that scores well here may still be slow, but a design that
scores badly here is definitively broken — no arrangement of speeds makes an
ingredient appear at a machine no belt reaches. A grader that errs must err
toward passing broken designs rather than failing working ones, or it teaches
the corpus builder to throw away good examples.

The metrics this yields — `delivers`, `fed`, `leaks`, `mixed` — are global in a
way the older ones are not: they cannot be satisfied by copying a local pattern,
only by getting the whole chain right. That is the point. A benchmark whose every
number is 0.98 has stopped measuring the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .blueprint import Blueprint
from .grammar import Port, Spec
from .prototypes import EAST, NORTH, SOUTH, WEST, Data, load

# Belt output direction, in tiles. Same table as `validate._AHEAD`; kept
# separate because these two modules are free to disagree about what a belt is
# and this one has to also know about splitters.
AHEAD = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}

# Inserters face the tile they insert *into* and pick up from the opposite side.
PICKUP = {NORTH: (0, 1), EAST: (-1, 0), SOUTH: (0, -1), WEST: (1, 0)}

BELTS = {"transport-belt", "underground-belt", "splitter"}

# How many item types one belt can carry. Two, because a belt has two lanes —
# the single most common way a hand-built factory jams.
BELT_LANES = 2

# The fixed point converges in at most one pass per entity; the cap is a
# guard against a cycle in the propagation rules rather than an approximation.
MAX_ROUNDS = 64


@dataclass
class Flow:
    """What reached where, and the four numbers that follow from it."""

    #: Output ports whose declared item actually arrives, over all output ports.
    delivers: float = 0.0
    #: Crafting machines that receive every ingredient of their recipe.
    fed: float = 0.0
    #: Crafting machines that are fed *and* whose product leaves them.
    working: float = 0.0
    #: Belts carrying more item types than a belt has lanes.
    mixed: int = 0
    #: Items flowing off the design at a tile no port declared.
    leaks: int = 0
    #: The design stayed inside the zone the spec asked for.
    within_zone: bool = True
    #: Per-item detail, for `--verbose` and for tests that need to say why.
    missing: tuple[str, ...] = ()
    carried: dict = field(default_factory=dict)

    def score(self) -> float:
        """Delivery is the headline; being fed is partial credit.

        Weighted rather than averaged because they are not independent: a module
        that delivers necessarily feeds its machines, so an unweighted mean would
        hand out 0.5 for the easy half of the problem.
        """
        return 0.7 * self.delivers + 0.3 * self.fed

    def to_dict(self) -> dict:
        out = asdict(self)
        out.pop("carried")
        out["missing"] = list(self.missing)
        return out


def trace(blueprint: Blueprint, spec: Spec | None = None, data: Data | None = None) -> Flow:
    """Propagate items from the input ports and report what arrives.

    With no spec, or a spec with no ports, the inputs are inferred: every belt
    on the boundary pointing inwards is treated as a source of whatever the
    machines it can reach want. That keeps the metric meaningful for the older
    generators, which have no ports, without pretending they declared any.
    """
    data = data or load()
    ports = spec.ports if spec else ()
    grid = _occupancy(blueprint, data)
    held: list[set[str]] = [set() for _ in blueprint.entities]

    for port in ports:
        if port.role != "in":
            continue
        for index in grid.get((port.x, port.y), ()):
            held[index].add(port.item)

    if not any(port.role == "in" for port in ports):
        _seed_from_edges(blueprint, data, grid, held)

    for _ in range(MAX_ROUNDS):
        if not _round(blueprint, data, grid, held):
            break

    return _report(blueprint, spec, data, grid, held, ports)


def _round(blueprint: Blueprint, data: Data, grid: dict, held: list[set[str]]) -> bool:
    """One propagation pass. Returns whether anything moved."""
    moved = False
    for index in range(len(blueprint.entities)):
        supply = _supply(blueprint, data, grid, index, held)
        if not supply:
            continue
        for target in _sinks(blueprint, data, grid, index):
            other = blueprint.entities[target]
            accepted = supply
            recipe = data.recipes.get(other.recipe or "")
            if recipe is not None:
                # A machine takes only what its recipe calls for. The inserter
                # would physically try to insert anything, and then hold it
                # forever, which is a jam rather than a delivery.
                accepted = supply & {name for name, _ in recipe.ingredients}
            if accepted - held[target]:
                held[target] |= accepted
                moved = True
    return moved


def _supply(
    blueprint: Blueprint, data: Data, grid: dict, index: int, held: list[set[str]]
) -> set[str]:
    """What this entity has available to hand on this round.

    An inserter is the one entity that does not carry its own stock: it reaches
    over to the tile behind it and takes whatever is there. Modelling it as a
    pull rather than as a push is what makes a machine's output reachable at all
    — nothing ever pushes items *into* an inserter.
    """
    placement = blueprint.entities[index]
    proto = data.entity(placement.name)
    if proto is None:
        return set()
    if proto.category != "inserter":
        return _made(blueprint, data, index, held)
    step = PICKUP.get(placement.direction)
    if step is None:
        return set()
    reach = int(proto.max_distance or 1)
    tile = (placement.x + step[0] * reach, placement.y + step[1] * reach)
    supply: set[str] = set()
    for source in grid.get(tile, ()):
        supply |= _made(blueprint, data, source, held)
    return supply


def _made(blueprint: Blueprint, data: Data, index: int, held: list[set[str]]) -> set[str]:
    """What this entity can hand on: its products if it crafts, else its load."""
    placement = blueprint.entities[index]
    recipe = data.recipes.get(placement.recipe or "")
    if recipe is None:
        return held[index]
    if not {name for name, _ in recipe.ingredients} <= held[index]:
        return set()
    return {name for name, _ in recipe.results}


def _sinks(blueprint: Blueprint, data: Data, grid: dict, index: int) -> list[int]:
    """The entities this one can put items into.

    Belts only ever feed belts: a belt running into an assembler in Factorio
    does nothing at all, and counting it as a delivery would let the model skip
    the inserters entirely and still score well.
    """
    placement = blueprint.entities[index]
    proto = data.entity(placement.name)
    if proto is None:
        return []
    if proto.category == "inserter":
        reach = int(proto.max_distance or 1)
        step = AHEAD.get(placement.direction)
        if step is None:
            return []
        tile = (placement.x + step[0] * reach, placement.y + step[1] * reach)
        return list(grid.get(tile, ()))
    if proto.category not in BELTS:
        return []
    if proto.category == "underground-belt" and placement.flow == "input":
        return _tunnel(blueprint, data, grid, index)
    step = AHEAD.get(placement.direction)
    if step is None:
        return []
    out: list[int] = []
    for tile in placement.tiles(data):
        ahead = (tile[0] + step[0], tile[1] + step[1])
        for target in grid.get(ahead, ()):
            other = data.entity(blueprint.entities[target].name)
            if other is not None and other.category in BELTS:
                out.append(target)
    return out


def _drains(blueprint: Blueprint, data: Data, grid: dict, index: int) -> bool:
    """Whether anything takes the product of the machine at `index` away.

    A fed assembler with no output inserter fills its output slot and stops, so
    "fed" alone would score a design that crafts exactly one batch.
    """
    tiles = set(blueprint.entities[index].tiles(data))
    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        if proto is None or proto.category != "inserter":
            continue
        step = PICKUP.get(placement.direction)
        if step is None:
            continue
        reach = int(proto.max_distance or 1)
        if (placement.x + step[0] * reach, placement.y + step[1] * reach) in tiles:
            return True
    return False


def _tunnel(blueprint: Blueprint, data: Data, grid: dict, index: int) -> list[int]:
    """The far end of an underground pair, if there is one.

    A tunnel is the one place where the next tile is *supposed* to be empty, so
    the belt rule above would report a dead end for a perfectly good pair.
    """
    placement = blueprint.entities[index]
    proto = data.entity(placement.name)
    step = AHEAD.get(placement.direction)
    if step is None or proto is None:
        return []
    for distance in range(1, int(proto.max_distance or 0) + 1):
        tile = (placement.x + step[0] * distance, placement.y + step[1] * distance)
        for target in grid.get(tile, ()):
            other = blueprint.entities[target]
            if (
                other.name == placement.name
                and other.direction == placement.direction
                and other.flow == "output"
            ):
                return [target]
    return []


def _seed_from_edges(blueprint: Blueprint, data: Data, grid: dict, held: list[set[str]]) -> None:
    """Ports for a design that never declared any.

    A belt on the boundary facing inwards is an input, and what it carries is
    guessed from what the design's machines need but nothing in the design makes.
    Guessed inputs make `fed` comparable across generators; they are never used
    to score `delivers`, which needs a declared promise to check against.
    """
    wanted, produced = set(), set()
    for placement in blueprint.entities:
        recipe = data.recipes.get(placement.recipe or "")
        if recipe is None:
            continue
        wanted |= {name for name, _ in recipe.ingredients}
        produced |= {name for name, _ in recipe.results}
    raws = wanted - produced
    if not raws:
        return
    x0, y0, x1, y1 = blueprint.bounds(data)
    for index, placement in enumerate(blueprint.entities):
        proto = data.entity(placement.name)
        if proto is None or proto.category not in BELTS:
            continue
        step = AHEAD.get(placement.direction)
        if step is None:
            continue
        behind = (placement.x - step[0], placement.y - step[1])
        if not (x0 <= behind[0] < x1 and y0 <= behind[1] < y1) and behind not in grid:
            held[index] |= raws


def _report(
    blueprint: Blueprint,
    spec: Spec | None,
    data: Data,
    grid: dict,
    held: list[set[str]],
    ports: tuple[Port, ...],
) -> Flow:
    outputs = [port for port in ports if port.role == "out"]
    arrived, missing = 0, []
    for port in outputs:
        if any(port.item in held[index] for index in grid.get((port.x, port.y), ())):
            arrived += 1
        else:
            missing.append(port.item)

    machines = [
        index
        for index, placement in enumerate(blueprint.entities)
        if data.recipes.get(placement.recipe or "") is not None
    ]
    fed = [index for index in machines if _made(blueprint, data, index, held)]
    working = [index for index in fed if _drains(blueprint, data, grid, index)]

    mixed, leaks = 0, 0
    declared = {(port.x, port.y) for port in outputs}
    x0, y0, x1, y1 = blueprint.bounds(data)
    for index, placement in enumerate(blueprint.entities):
        proto = data.entity(placement.name)
        if proto is None or proto.category not in BELTS:
            continue
        if len(held[index]) > BELT_LANES:
            mixed += 1
        step = AHEAD.get(placement.direction)
        if step is None or not held[index]:
            continue
        ahead = (placement.x + step[0], placement.y + step[1])
        outside = not (x0 <= ahead[0] < x1 and y0 <= ahead[1] < y1)
        if outside and (placement.x, placement.y) not in declared:
            leaks += 1

    within = True
    if spec is not None and spec.ports:
        width, height = blueprint.extent(data)
        within = width <= spec.width and height <= spec.height

    return Flow(
        delivers=arrived / len(outputs) if outputs else 0.0,
        fed=len(fed) / len(machines) if machines else 0.0,
        working=len(working) / len(machines) if machines else 0.0,
        mixed=mixed,
        leaks=leaks,
        within_zone=within,
        missing=tuple(dict.fromkeys(missing)),
        carried={index: tuple(sorted(items)) for index, items in enumerate(held) if items},
    )


def _occupancy(blueprint: Blueprint, data: Data) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = {}
    for index, placement in enumerate(blueprint.entities):
        for tile in placement.tiles(data):
            grid.setdefault(tile, []).append(index)
    return grid
