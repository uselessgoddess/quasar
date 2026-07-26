"""The recipe arithmetic: what a module has to contain before anything is placed.

This is the half of "build me electronic circuits from iron and copper plate"
that is not a spatial problem at all. Copper plate becomes copper cable, cable
and iron plate become circuits, and three cable machines feed two circuit
machines because the recipes say so. None of that is worth a single parameter of
a neural network: `assets/prototypes.json` already states every ingredient list,
craft time and machine speed exactly, and a planner that reads them is right by
construction where a model would be plausibly wrong — a mis-remembered ratio
produces a factory that looks perfect and starves.

So the split this module exists to make is a compiler's: the planner is the
front end and decides *what* to build, the model is the back end and decides
*where it goes*. The plan then rides in the prompt (see `grammar.Step`), which
means the model is conditioned on the ratios rather than asked to invent them —
and, because the plan is written into the training document too, a later and
larger member of the family can be asked to emit it and be scored against the
planner's answer.

Two deliberate limits:

*Chains stop at the declared supply.* `solve` expands a product until every leaf
is something the caller said would arrive on a belt. An item that is neither
supplied nor craftable from what is raises `PlanError` rather than quietly
inventing a mine.

*Ratios are integers of machines, not fractions.* Two thirds of an assembler is
not a thing that can be placed, so counts round up and `Stage.rate` reports what
the rounded-up count actually produces. That is the number a throughput metric
should compare against, not the fractional ideal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .grammar import Step
from .prototypes import Data, Recipe, load

# Machine tiers, cheapest first, per crafting category. The planner picks by
# explicit tier rather than by asking for the fastest crafter, so a plan is
# reproducible and an early-game module stays early-game.
CRAFTERS = {
    "crafting": ("assembling-machine-1", "assembling-machine-2", "assembling-machine-3"),
    "advanced-crafting": ("assembling-machine-2", "assembling-machine-2", "assembling-machine-3"),
    "crafting-with-fluid": ("assembling-machine-2", "assembling-machine-2", "assembling-machine-3"),
    "smelting": ("stone-furnace", "steel-furnace", "electric-furnace"),
    "chemistry": ("chemical-plant", "chemical-plant", "chemical-plant"),
    "oil-processing": ("oil-refinery", "oil-refinery", "oil-refinery"),
    "centrifuging": ("centrifuge", "centrifuge", "centrifuge"),
}

# How deep a chain may go before the planner gives up. Three is enough for
# plate -> cable -> circuit and for plate -> gear -> belt; a module deeper than
# that is a factory, not a module.
MAX_DEPTH = 4


class PlanError(ValueError):
    """The chain cannot be built from what the caller said it would be given."""


@dataclass(frozen=True)
class Stage:
    """One recipe, the machine that runs it, and how many of them.

    Ordered by the plan, not by preference: stage *n* only ever consumes the
    supply or the products of stages before it, which is what lets a layout put
    them in a line and let each row feed the next.
    """

    recipe: str
    machine: str
    count: int
    #: Items per second of `product` the rounded-up machine count really makes.
    rate: float
    product: str
    ingredients: tuple[str, ...]

    def step(self) -> Step:
        return Step(machine=self.machine, recipe=self.recipe, count=self.count)


@dataclass(frozen=True)
class Plan:
    """A whole module, as arithmetic: supply in, stages, product out."""

    product: str
    supply: tuple[str, ...]
    stages: tuple[Stage, ...]

    @property
    def machines(self) -> int:
        return sum(stage.count for stage in self.stages)

    @property
    def rate(self) -> float:
        """Items per second of the product, at the last stage."""
        return self.stages[-1].rate if self.stages else 0.0

    def steps(self) -> tuple[Step, ...]:
        return tuple(stage.step() for stage in self.stages)

    def raws_of(self, stage: Stage) -> tuple[str, ...]:
        """The ingredients of `stage` that arrive from outside rather than from
        another stage — which is exactly the set a layout has to route a belt for.
        """
        return tuple(item for item in stage.ingredients if item in self.supply)


def machine_for(data: Data, recipe: Recipe, tier: int = 0) -> str:
    """The machine this planner runs `recipe` in, at `tier`.

    Falls back to the table rather than to a guess: a category the tier list
    does not name is looked up through `crafters_for`, which is derived from the
    prototypes and cannot drift from them.
    """
    names = CRAFTERS.get(recipe.category)
    if names is not None:
        return names[max(0, min(tier, len(names) - 1))]
    able = data.crafters_for(recipe)
    if not able:
        raise PlanError(f"nothing can craft {recipe.name!r} ({recipe.category})")
    return able[max(0, min(tier, len(able) - 1))].name


def is_fluid(data: Data, item: str) -> bool:
    """Fluids are the items with no stack size — they need pipes, not belts."""
    return item not in data.stack_sizes


def belted(data: Data, item: str) -> bool:
    """Whether a module treats `item` as arriving rather than as something it makes.

    The boundary is not arbitrary: a furnace has no recipe slot in a blueprint —
    it picks its recipe from whatever is inserted — so a smelting step cannot be
    specified in this grammar at all, and a plate is therefore the natural thing
    for a module to be handed. Ore is on the far side of that line, which is why
    "circuits from plates" is a module and "circuits from ore" is a factory.
    """
    if is_fluid(data, item):
        return False
    recipe = recipe_for(data, item)
    if recipe is None:
        return True
    return not any(machine.takes_recipe for machine in data.crafters_for(recipe))


def recipe_for(data: Data, item: str) -> Recipe | None:
    """The recipe this planner uses to make `item`, or `None` if it is raw.

    `producers_of` biases toward the shortest ingredient list, which is the
    basic route rather than the advanced one. Recipes that merely repackage the
    item — barrelling, and anything consuming the item it produces — are skipped
    so that a chain cannot loop back on itself.
    """
    for recipe in data.producers_of(item):
        if any(name == item for name, _ in recipe.ingredients):
            continue
        return recipe
    return None


def solve(
    data: Data | None,
    product: str,
    supply,
    *,
    rate: float = 1.0,
    tier: int = 0,
    depth: int = MAX_DEPTH,
) -> Plan:
    """Expand `product` into stages, stopping at `supply`.

    `rate` is items per second of the finished product. Demand flows backwards
    through the chain, so a stage's machine count is decided by everything that
    consumes it, and the stages come back in build order: supply first.
    """
    data = data or load()
    supply = tuple(dict.fromkeys(supply))
    for item in (product, *supply):
        if item not in data.stack_sizes and item not in data.recipes:
            raise PlanError(f"unknown item {item!r}")

    demand: dict[str, float] = {}
    order: list[str] = []

    def walk(item: str, wanted: float, level: int, path: tuple[str, ...]) -> None:
        if item in supply:
            return
        if item in path:
            raise PlanError(f"{item!r} is made from itself")
        if level > depth:
            raise PlanError(f"{product!r} needs more than {depth} stages from {list(supply)}")
        recipe = recipe_for(data, item)
        if recipe is None:
            raise PlanError(f"{item!r} is not craftable and was not supplied")
        seen = demand.get(item)
        demand[item] = (seen or 0.0) + wanted
        if seen is None:
            order.append(item)
        crafts = wanted / recipe.yields(item)
        for name, amount in recipe.ingredients:
            walk(name, crafts * amount, level + 1, (*path, item))

    walk(product, rate, 0, ())
    if not order:
        raise PlanError(f"{product!r} was already supplied")

    stages = []
    # Deepest first: an item is only ever appended after everything that wanted
    # it, so reversing discovery order is a topological sort of the chain.
    for item in reversed(order):
        recipe = recipe_for(data, item)
        assert recipe is not None  # walk would have raised
        machine = machine_for(data, recipe, tier)
        speed = data.entities[machine].crafting_speed or 1.0
        per_machine = speed / recipe.time * recipe.yields(item)
        count = max(1, math.ceil(demand[item] / per_machine - 1e-9))
        stages.append(
            Stage(
                recipe=recipe.name,
                machine=machine,
                count=count,
                rate=count * per_machine,
                product=item,
                ingredients=tuple(name for name, _ in recipe.ingredients),
            )
        )
    return Plan(product=product, supply=supply, stages=tuple(stages))


def fit(
    data: Data | None,
    product: str,
    supply,
    *,
    machines: int,
    tier: int = 0,
    depth: int = MAX_DEPTH,
) -> Plan:
    """The largest plan whose machine count fits in `machines`.

    A module is budgeted by the zone the player marked out rather than by a
    throughput target: "as much as fits in here" is the request, and the rate
    that comes out is the answer. Scaling is searched rather than solved because
    the per-stage rounding is not linear — three stages at 1.5 machines each
    fit in five, and doubling them needs nine, not ten.
    """
    data = data or load()
    unit = solve(data, product, supply, rate=1e-9, tier=tier, depth=depth)
    if unit.machines > machines:
        raise PlanError(f"{product!r} needs {unit.machines} machines, budget is {machines}")

    best, step = unit, unit.rate
    if step <= 0:
        return best
    for scale in range(2, machines + 2):
        candidate = solve(data, product, supply, rate=step * scale, tier=tier, depth=depth)
        if candidate.machines > machines:
            break
        best = candidate
    return best


def chains(
    data: Data | None = None, *, depth: int = 3, tier: int = 0
) -> dict[str, tuple[str, ...]]:
    """Every product this planner can build from belt-able items, and from what.

    Used by the module generator to draw targets, and by the tests to assert
    that the catalogue is not empty. The supply of a product is the set of
    leaves its own chain bottoms out at: `electronic-circuit` maps to
    `(iron-plate, copper-plate)`, which is exactly the pair of input ports a
    module for it needs.
    """
    data = data or load()
    out: dict[str, tuple[str, ...]] = {}
    for _, recipe in sorted(data.recipes.items()):
        for item, _ in recipe.results:
            if is_fluid(data, item) or item in out:
                continue
            leaves = _leaves(data, item, depth)
            if leaves is None:
                continue
            try:
                solve(data, item, leaves, rate=1.0, tier=tier, depth=depth)
            except PlanError:
                continue
            out[item] = leaves
    return out


@dataclass(frozen=True)
class Module:
    """One entry in the module catalogue: what to build, from what, how deep.

    The depth travels with the pair because it is not a preference — it is part
    of the identity of the module. `underground-belt` from plate and gears is a
    two-stage design; `underground-belt` from plate alone is a three-stage one
    that makes its own gears. Solving the second at the first's depth raises, and
    solving the first at the second's depth silently returns the first, so a
    caller that drops this field draws targets it cannot build.
    """

    product: str
    supply: tuple[str, ...]
    depth: int


def modules(
    data: Data | None = None,
    *,
    depth: int = 3,
    lanes: int = 2,
) -> tuple[Module, ...]:
    """The chains a stacked-band module layout can actually build, and from what.

    Four filters, each of them a property of the layout rather than of the
    recipe: every stage must run in a machine that has a recipe slot (a furnace
    does not), the chain must be at least two stages long (one stage is an
    assembler row, which the corpus already has), each stage's shared belt must
    carry no more than `lanes` item types, and the chain must be *linear*.

    Linear is the sharp one. A stacked-band layout hands a stage's product to
    the row immediately below it and nowhere else, so a chain where stage one's
    product is wanted again three stages down cannot be built by stacking:
    `fast-underground-belt` needs both gears and yellow undergrounds, and the
    gears would sail past the underground row with nothing to take them off. The
    grader in `flow` catches exactly this — `fed` at 0.67, the last stage
    starved — which is how the condition was found rather than guessed. Routing
    an intermediate down past a stage is a real layout and a later generator can
    have it; declaring it out of scope here is what keeps the corpus honest.

    Every boundary from two stages up to `depth` is offered, because they are
    different modules and not different qualities of the same one: a player who
    has gears on the bus wants the two-stage belt module, and one who has only
    plate wants the three-stage one.
    """
    data = data or load()
    out: dict[tuple[str, tuple[str, ...]], Module] = {}
    for level in range(2, depth + 1):
        for product, supply in chains(data, depth=level).items():
            try:
                unit = solve(data, product, supply, rate=1e-9, depth=level)
            except PlanError:
                continue
            if len(unit.stages) < 2:
                continue
            if any(not data.entities[stage.machine].takes_recipe for stage in unit.stages):
                continue
            if any(
                len(unit.raws_of(stage)) + bool(rank) > lanes
                for rank, stage in enumerate(unit.stages)
            ):
                continue
            if not _linear(unit):
                continue
            used = _used(unit)
            out.setdefault((product, used), Module(product, used, level))
    return tuple(out.values())


def _used(unit: Plan) -> tuple[str, ...]:
    """The supply the plan actually consumes.

    `_leaves` can name an item that the solved plan then never asks for:
    `arithmetic-combinator` reaches copper cable by two paths of different
    lengths, so cable is both expanded from copper plate on the short path and
    declared raw on the long one, and the plan that stops at cable leaves the
    plate unused. Handing that supply to the generator would declare an input
    port for an item no machine takes — a leak by construction, and the model
    would learn that a port in the prompt need not be connected to anything.
    """
    wanted = {item for stage in unit.stages for item in stage.ingredients}
    return tuple(item for item in unit.supply if item in wanted)


def _linear(unit: Plan) -> bool:
    """Whether every stage is fed by the supply and by the stage just above it."""
    reachable = set(unit.supply)
    for stage in unit.stages:
        if not set(stage.ingredients) <= reachable:
            return False
        reachable = set(unit.supply) | {stage.product}
    return True


def _leaves(data: Data, item: str, depth: int) -> tuple[str, ...] | None:
    """The belt-able raw materials `item` bottoms out at, or `None` if it cannot.

    `None` covers everything a module is not allowed to need: a fluid anywhere
    in the chain, a loop, or a chain deeper than `depth`.
    """
    found: list[str] = []

    def walk(name: str, level: int, path: tuple[str, ...]) -> bool:
        if is_fluid(data, name) or name in path:
            return False
        recipe = recipe_for(data, name)
        if belted(data, name) or level >= depth:
            # Past the depth limit an intermediate is simply declared to arrive
            # on a belt, which is what a module boundary is: `advanced-circuit`
            # from plates is a factory, `advanced-circuit` from circuits,
            # plastic and cable is a module.
            if name not in found:
                found.append(name)
            return True
        assert recipe is not None  # `belted` is true for anything with no recipe
        return all(
            walk(ingredient, level + 1, (*path, name)) for ingredient, _ in recipe.ingredients
        )

    return tuple(found) if walk(item, 0, ()) and found else None
