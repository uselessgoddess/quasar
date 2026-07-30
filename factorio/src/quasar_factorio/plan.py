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

# How deep a chain may go before the planner gives up. Four reaches the first
# factory-sized graph, green science: six recipes, a shared gear intermediate,
# and two routes that converge on the final science assembler.
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
    """Whether `item` needs a pipe rather than a belt.

    Asked of the fluid table first and of the stack sizes only as a fallback,
    because the fallback alone was a bug with a long reach: `data.raw.item`
    holds gears and plates, but a science pack is a `tool` and a piercing round
    is `ammo`, so 61 of the items vanilla recipes name had no stack size and
    every one of them was called a fluid. That silently deleted the entire
    science branch from `chains` — the planner did not refuse to build green
    science, it never saw it — which is why the catalogue looked like proof that
    branching chains were the obstacle. Unknown names still count as fluids so
    that a typo fails closed rather than being routed onto a belt.
    """
    return item in data.fluids or item not in data.stack_sizes


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
        crafts = wanted / recipe.yields(item)
        for name, amount in recipe.ingredients:
            walk(name, crafts * amount, level + 1, (*path, item))
        if seen is None:
            # Recorded on the way *out*, not on the way in. Order-of-discovery
            # reversed is not a topological sort when the graph has cross edges
            # of different lengths: an arithmetic combinator names cable and
            # circuit as ingredients in that order, discovers cable first, and
            # so reversed discovery puts the circuit stage above the cable stage
            # that feeds it. Appending after the recursion puts every ingredient
            # in front of the thing it goes into, whatever the shape.
            order.append(item)

    walk(product, rate, 0, ())
    if not order:
        raise PlanError(f"{product!r} was already supplied")

    stages = []
    for item in order:
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

    `shape` says which layout the chain is for: `"stack"` is the run of bands
    that each hand their product to the row below, `"fork"` is two such runs side
    by side converging on a last machine that consumes both, and `"factory"` is
    an explicitly supported DAG with shared intermediates and multiple input
    belts. It is defaulted because a stacked module was the original catalogue
    and the pair-plus-depth remains its identity. Generic shapes are derived
    from the plan; factory shapes are admitted only beside a concrete generator,
    so the catalogue never promises geometry it cannot produce.
    """

    product: str
    supply: tuple[str, ...]
    depth: int
    shape: str = "stack"


def modules(
    data: Data | None = None,
    *,
    depth: int = 4,
    lanes: int = 2,
) -> tuple[Module, ...]:
    """The chains a module layout can actually build, and from what.

    Three filters and a shape. The filters are properties of the layout rather
    than of the recipe: every stage must run in a machine that has a recipe slot
    (a furnace does not), the chain must be at least two stages long (one stage
    is an assembler row, which the corpus already has), and no belt may carry
    more than `lanes` item types. The shape is which supported module layout can
    express the plan, and a chain none of them fits is dropped.

    *Stack* is a run of bands: each stage's belt carries what arrives from
    outside plus the product of the stage immediately above, and nothing else.
    That is the sharp condition, because a stacked band hands its product down
    one row and nowhere else — `fast-underground-belt` needs both gears and
    yellow undergrounds, and the gears would sail past the underground row with
    nothing to take them off. The grader in `flow` catches exactly this — `fed`
    at 0.67, the last stage starved — which is how the condition was found
    rather than guessed.

    *Fork* is the answer to that, for the case where the two wanted
    intermediates are made rather than delivered: two stacks side by side, each
    linear in its own right, both dropping onto one belt that the last machine
    picks up from. It buys the branching chains — a boiler from stone furnaces
    and pipes, a repair pack from circuits and gears — at the price of a
    convergence belt that is full: two branch products are already `lanes` item
    types, so the last stage may consume those two and nothing else. A chain
    that additionally wants a raw item at the bottom cannot fit this shape.

    *Factory* is deliberately concrete rather than a claim that arbitrary DAG
    placement is solved. Green science is the first admitted example: gears are
    shared by the inserter and transport-belt branches, while the inserter needs
    circuits, gears, and iron plate. Power switch is the second: cable feeds both
    circuits and the final three-ingredient stage. Fast splitter is the third
    and first double diamond: circuits and gears are both shared, and both the
    splitter and fast-splitter stages join three ingredients. Their generators
    route those cross-edges explicitly instead of pretending the generic stack
    can.

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
            shape = _shape(unit, lanes)
            if shape is None:
                continue
            used = _used(unit)
            out.setdefault((product, used), Module(product, used, level, shape))

    # Factory-sized layouts are deliberately explicit.  Their plans contain
    # cross-edges and three-item stages that the one-belt stack and disjoint
    # fork cannot express.  Keeping this gate beside the generic filters makes
    # the supported geometry honest: an arbitrary DAG does not enter training
    # merely because the planner can count it.
    if lanes >= 2:
        supply = ("iron-plate", "copper-plate")
        factories = (
            ("power-switch", 3),
            ("fast-splitter", 4),
            ("logistic-science-pack", 4),
        )
        for product, required_depth in factories:
            if depth < required_depth:
                continue
            unit = solve(data, product, supply, rate=1e-9, depth=required_depth)
            if all(data.entities[stage.machine].takes_recipe for stage in unit.stages):
                out[(product, supply)] = Module(product, supply, required_depth, "factory")
    return tuple(out.values())


def fork(unit: Plan, lanes: int = 2) -> tuple[tuple[Stage, ...], tuple[Stage, ...]] | None:
    """The plan split into two branches converging on its last stage, or `None`.

    A plan forks when its last stage consumes exactly two items, both of them
    made by earlier stages, and those earlier stages fall into two disjoint runs
    that each stack on their own. The two runs come back in the plan's own order,
    which is a topological one, so a layout can place each branch as a column of
    bands and be sure a stage is never drawn above something it consumes.

    The last stage is deliberately not returned: it is `unit.stages[-1]`, it is
    always the whole of the third part, and returning it would invite a caller to
    treat the three pieces as interchangeable when only the two branches are.
    """
    stages = unit.stages
    if len(stages) < 3 or lanes < 2:
        return None
    made = {stage.product: stage for stage in stages[:-1]}
    heads = tuple(dict.fromkeys(stages[-1].ingredients))
    # Exactly two, both made here: one made item is a stack, three would not fit
    # on the convergence belt, and a raw item among them would need a lane the
    # two branch products have already taken.
    if len(heads) != 2 or any(item not in made for item in heads):
        return None
    left, right = (_ancestry(stages[:-1], head, made) for head in heads)
    if len(left) + len(right) != len(stages) - 1 or set(left) & set(right):
        return None
    if not (_banded(unit, left, lanes) and _banded(unit, right, lanes)):
        return None
    return left, right


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


def _shape(unit: Plan, lanes: int) -> str | None:
    """Which module layout can express this plan, or `None` if neither can.

    The two are mutually exclusive rather than merely tried in order: a stack
    reaches its last stage with one made item in hand, a fork with two, so no
    plan is both.
    """
    if _banded(unit, unit.stages, lanes):
        return "stack"
    if fork(unit, lanes) is not None:
        return "fork"
    return None


def _banded(unit: Plan, stages: tuple[Stage, ...], lanes: int) -> bool:
    """Whether `stages` can be one column of bands: linear, and `lanes` per belt.

    The belt above a band carries what the stage takes from outside the module,
    plus — for every band but the first — the product of the band above it. Both
    conditions are checked here because they are the same condition seen twice:
    a stage fed by something other than the supply and the row above has no belt
    to be fed from, and a stage fed by too many things has no room on the one it
    has.
    """
    reachable = set(unit.supply)
    for rank, stage in enumerate(stages):
        if not set(stage.ingredients) <= reachable:
            return False
        if len(unit.raws_of(stage)) + bool(rank) > lanes:
            return False
        reachable = set(unit.supply) | {stage.product}
    return True


def _ancestry(stages: tuple[Stage, ...], head: str, made: dict[str, Stage]) -> tuple[Stage, ...]:
    """The stages that `head` is built out of, `head` included, in plan order."""
    wanted = {head}
    frontier = [head]
    while frontier:
        stage = made.get(frontier.pop())
        if stage is None:
            continue
        for item in stage.ingredients:
            if item in made and item not in wanted:
                wanted.add(item)
                frontier.append(item)
    return tuple(stage for stage in stages if stage.product in wanted)


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
