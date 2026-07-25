"""Synthetic blueprint generators: the backbone of the corpus.

There are only a few thousand small, clean, vanilla blueprints on the public
sites, and a 3M-parameter model needs tens of thousands of examples before the
grammar stops being the hard part. So most of the corpus is generated from
parameterised versions of the layouts every Factorio player builds by hand — a
smelter column, an assembler row, a drill outpost — with the tier, length,
direction, spacing and recipe randomised.

The generators are held to the same standard as the model: every one of them
builds through :class:`Canvas`, which refuses an overlapping placement, and the
test suite asserts that a thousand random draws from every generator are valid
*and* score well on connectivity. A generator that emitted subtly broken
factories would teach the model to emit subtly broken factories.

Geometry notes that are easy to get backwards, and are relied on here: an
inserter *faces the tile it inserts into* and picks up from the opposite side; a
belt moves items toward the tile it faces; a medium electric pole covers 7x7 and
reaches 9 tiles, a small one covers 5x5 and reaches 7.5.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Callable
from dataclasses import dataclass

from .blueprint import Blueprint, Placement, rotate
from .grammar import Spec
from .prototypes import EAST, NORTH, SOUTH, WEST, Data, load

# Belt, inserter and furnace tiers, cheapest first. Sampling a tier per
# blueprint rather than per entity is what keeps a generated design coherent —
# real bases do not mix yellow and blue belt at random inside one column.
BELTS = ("transport-belt", "fast-transport-belt", "express-transport-belt")
UNDERGROUNDS = ("underground-belt", "fast-underground-belt", "express-underground-belt")
SPLITTERS = ("splitter", "fast-splitter", "express-splitter")
# Reach-1 inserters only. A long-handed inserter reaches two tiles and needs a
# layout built around that; dropping one into a reach-1 slot produces a blueprint
# that looks fine and picks up from thin air.
INSERTERS = ("inserter", "fast-inserter", "stack-inserter")
FURNACES = ("stone-furnace", "steel-furnace", "electric-furnace")
ASSEMBLERS = ("assembling-machine-1", "assembling-machine-2", "assembling-machine-3")
CHESTS = ("wooden-chest", "iron-chest", "steel-chest")

# Recipes worth putting in an assembler row: everything an ordinary base builds
# a dedicated line for, restricted to solid-only ingredients so the layout does
# not silently need a pipe it has not got.
SOLID_RECIPES = (
    "electronic-circuit",
    "advanced-circuit",
    "iron-gear-wheel",
    "copper-cable",
    "transport-belt",
    "underground-belt",
    "splitter",
    "inserter",
    "fast-inserter",
    "long-handed-inserter",
    "electric-mining-drill",
    "assembling-machine-1",
    "assembling-machine-2",
    "steel-chest",
    "iron-chest",
    "pipe",
    "pipe-to-ground",
    "engine-unit",
    "electric-furnace",
    "small-electric-pole",
    "medium-electric-pole",
    "big-electric-pole",
    "radar",
    "repair-pack",
    "automation-science-pack",
    "logistic-science-pack",
    "military-science-pack",
    "firearm-magazine",
    "piercing-rounds-magazine",
    "grenade",
    "wall",
    "rail",
    "solar-panel",
    "accumulator",
    "battery",
    "productivity-module",
    "speed-module",
    "efficiency-module",
)

SMELTED = ("iron-plate", "copper-plate", "steel-plate", "stone-brick")

CHEMICALS = ("plastic-bar", "sulfur", "sulfuric-acid", "battery", "explosives")


class Collision(AssertionError):
    """A generator tried to place two entities on the same tile.

    Never caught: it means a template's arithmetic is wrong, and the right
    response is to fix the template rather than to ship a broken example.
    """


@dataclass
class Canvas:
    """A blueprint under construction that refuses to overlap itself."""

    data: Data
    entities: list[Placement]
    taken: dict[tuple[int, int], str]

    @classmethod
    def new(cls, data: Data | None = None) -> Canvas:
        return cls(data=data or load(), entities=[], taken={})

    def free(self, name: str, x: int, y: int, direction: int = NORTH) -> bool:
        proto = self.data.entities[name]
        width, height = proto.footprint(direction)
        return all(
            (x + dx, y + dy) not in self.taken for dy in range(height) for dx in range(width)
        )

    def place(
        self,
        name: str,
        x: int,
        y: int,
        direction: int = NORTH,
        recipe: str | None = None,
        flow: str | None = None,
    ) -> Placement:
        proto = self.data.entities[name]
        width, height = proto.footprint(direction)
        for dy in range(height):
            for dx in range(width):
                tile = (x + dx, y + dy)
                if tile in self.taken:
                    raise Collision(f"{name} at {tile} would sit on {self.taken[tile]}")
                self.taken[tile] = name
        placement = Placement(
            name=name,
            x=x,
            y=y,
            direction=direction,
            recipe=recipe if proto.takes_recipe else None,
            flow=(flow or "input") if proto.takes_flow else None,
        )
        self.entities.append(placement)
        return placement

    def maybe(self, name: str, x: int, y: int, direction: int = NORTH) -> bool:
        """Place only if the tiles are free. For decorations, never structure."""
        if not self.free(name, x, y, direction):
            return False
        self.place(name, x, y, direction)
        return True

    def line(self, name: str, x: int, y: int, length: int, direction: int) -> None:
        """A run of `length` belts flowing toward `direction`."""
        step = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}[direction]
        for index in range(length):
            self.place(name, x + step[0] * index, y + step[1] * index, direction)

    def build(self, label: str = "") -> Blueprint:
        return Blueprint(entities=list(self.entities), label=label, source="synth").normalised(
            self.data
        )


Generator = Callable[[random.Random, Data], tuple[Blueprint, Spec]]


@functools.cache
def _solid(data: Data) -> tuple[str, ...]:
    """`SOLID_RECIPES` intersected with what this Factorio version actually has.

    The list is written by hand from the wiki, and the wiki and the prototype
    dump disagree about a few names (`effectivity-module`, not
    `efficiency-module`). The dump wins.
    """
    return tuple(name for name in SOLID_RECIPES if name in data.recipes)


@functools.cache
def _makeable(data: Data, machine: str) -> tuple[str, ...]:
    """The solid recipes `machine` is actually allowed to run.

    Asking first is better than picking blind and patching afterwards: an
    assembling machine 1 has no fluid boxes, so a `crafting-with-fluid` recipe
    is not merely slow in it, it is unbuildable.
    """
    categories = data.entities[machine].crafting_categories
    return tuple(name for name in _solid(data) if data.recipes[name].category in categories)


def _tier(rng: random.Random) -> int:
    """Belt/machine tier, biased low. Most real blueprints are yellow-belt."""
    return rng.choices((0, 1, 2), weights=(5, 3, 2))[0]


def belt_lane(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A straight belt run, optionally with an underground gap and a splitter.

    The simplest thing in the corpus, and the one that teaches the model that
    consecutive belts share a direction — which is most of what belt sanity is.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt, under, split = BELTS[tier], UNDERGROUNDS[tier], SPLITTERS[tier]
    lanes = rng.randint(1, 4)
    # Capped so a lane bundle stays inside the sequence budget: five tokens an
    # entity means ~80 entities is the most a 512-token context will hold.
    length = rng.randint(6, min(40, 72 // lanes))
    gap = rng.choice((0, 0, 1))

    # A splitter spans two adjacent lanes, so its tiles are reserved before the
    # belts are laid rather than carved out afterwards.
    splitter_at, splitter_lane = None, None
    if lanes >= 2 and gap == 0 and rng.random() < 0.4:
        splitter_at = rng.randrange(1, max(2, length - 1))
        splitter_lane = rng.randrange(lanes - 1)
        canvas.place(split, splitter_at, splitter_lane, EAST)

    for lane in range(lanes):
        y = lane * (1 + gap)
        x = 0
        while x < length:
            if (
                splitter_at is not None
                and x == splitter_at
                and lane
                in (
                    splitter_lane,
                    splitter_lane + 1,
                )
            ):
                x += 1
                continue
            # An underground pair skips up to `max_distance` tiles; the entrance
            # faces the flow and the exit is flipped, which is how the game
            # spells "output".
            hop = rng.randint(2, int(data.entities[under].max_distance or 5))
            if (
                x
                and x + hop + 1 < length
                and rng.random() < 0.12
                and all(canvas.free(under, x + step, y, EAST) for step in (0, hop))
            ):
                canvas.place(under, x, y, EAST, flow="input")
                canvas.place(under, x + hop, y, EAST, flow="output")
                x += hop + 1
            else:
                if canvas.free(belt, x, y, EAST):
                    canvas.place(belt, x, y, EAST)
                x += 1

    blueprint = canvas.build("belt lane")
    return _oriented(blueprint, rng, data, "belt-lane", None)


def smelter_column(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """The classic two-row smelter: belt, inserters, furnaces, inserters, belt.

    Rows, top to bottom: input belt at y=0; input inserters facing south at y=1;
    the furnaces; output inserters facing south; output belt. The pitch is the
    furnace's own width, because a stone furnace is 2x2 and an electric one is
    3x3 — hard-coding either would silently break the other tier.

    Poles go in the spare column of the inserter row, spaced at twice the pitch,
    which stays inside a medium pole's 9-tile wire reach and covers both
    inserter rows with its 7x7 supply area.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt = BELTS[tier]
    furnace = FURNACES[tier]
    inserter = INSERTERS[min(tier, len(INSERTERS) - 1)]
    pitch, depth = data.entities[furnace].width, data.entities[furnace].height
    # Counting the belts and inserters, each furnace costs about seven and a half
    # entities at 2x2 and nine and a half at 3x3. The cap keeps the longest
    # column inside a 512-token context.
    count = rng.randint(3, 12 if pitch == 2 else 9)
    width = count * pitch
    out_row = 2 + depth

    canvas.line(belt, 0, 0, width, EAST)
    canvas.line(belt, 0, out_row + 1, width, EAST)
    for index in range(count):
        x = index * pitch
        canvas.place(furnace, x, 2)
        canvas.place(inserter, x, 1, SOUTH)
        canvas.place(inserter, x, out_row, SOUTH)
    for x in range(1, width, 2 * pitch):
        canvas.maybe("medium-electric-pole", x, 1)

    product = rng.choice(SMELTED[: 3 if tier else 2])
    blueprint = canvas.build(f"{count}x {furnace}")
    return _oriented(blueprint, rng, data, "smelter-column", product)


def assembler_row(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A row of assemblers fed from one belt and unloading onto another.

    Assemblers are 3x3, so the inserter rows have two spare columns per machine
    and the poles go in the first of them, six tiles apart — the largest spacing
    whose 7x7 supply still reaches the far assembler and the output inserters.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt = BELTS[tier]
    machine = ASSEMBLERS[tier]
    inserter = INSERTERS[min(tier + 1, len(INSERTERS) - 1)]
    # Tier 1 assemblers cannot run every category, so the machine is promoted
    # until it can rather than the recipe being quietly swapped: a row of
    # assembling machine 2s making electronic circuits is a real design, a row of
    # machine 1s set to a fluid recipe is one Factorio refuses to place.
    if not _makeable(data, machine):
        machine = "assembling-machine-2"
    recipe = rng.choice(_makeable(data, machine))

    count = rng.randint(2, 9)
    width = count * 3

    canvas.line(belt, 0, 0, width, EAST)
    canvas.line(belt, 0, 6, width, EAST)
    for index in range(count):
        x = index * 3
        canvas.place(machine, x, 2, recipe=recipe)
        canvas.place(inserter, x, 1, SOUTH)
        canvas.place(inserter, x, 5, SOUTH)
    for x in range(1, width, 6):
        canvas.maybe("medium-electric-pole", x, 1)

    blueprint = canvas.build(f"{count}x {recipe}")
    return _oriented(blueprint, rng, data, "assembler-row", recipe)


def mall_cell(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """One machine, a supply belt and a chest: the unit a mall repeats.

    Small on purpose. Short documents are where an autoregressive model first
    learns that `</bp>` exists, and a corpus of only large designs trains a model
    that never stops.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    machine = ASSEMBLERS[max(tier, 1)]
    inserter = INSERTERS[min(tier + 1, len(INSERTERS) - 1)]
    recipe = rng.choice(_makeable(data, machine))

    canvas.line(BELTS[tier], 0, 0, 3, EAST)
    canvas.place(inserter, 0, 1, SOUTH)
    canvas.place(machine, 0, 2, recipe=recipe)
    canvas.place(inserter, 0, 5, SOUTH)
    canvas.place(rng.choice(CHESTS[: tier + 2]), 0, 6)
    canvas.maybe("medium-electric-pole", 2, 1)

    blueprint = canvas.build(recipe)
    return _oriented(blueprint, rng, data, "mall-cell", recipe)


def mining_outpost(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """Two facing rows of drills unloading onto a shared belt.

    Drills are 3x3 and drop onto the tile they face, so both rows point at the
    belt between them. Every third column trades its upper drill for a medium
    pole: nine tiles apart is exactly the medium pole's wire reach, and its 7x7
    supply from that slot still touches the drill row on the far side of the
    belt, so one pole line powers both rows.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt = BELTS[tier]
    # A multiple of three: the pole slot is every third column, so any other
    # count leaves a tail of drills past the last pole with nothing to power them.
    columns = 3 * rng.randint(1, 4)
    width = columns * 3

    canvas.line(belt, 0, 3, width, EAST)
    for index in range(columns):
        x = index * 3
        if index % 3 == 1:
            canvas.place("medium-electric-pole", x + 1, 1)
        else:
            canvas.place("electric-mining-drill", x, 0, SOUTH)
        canvas.place("electric-mining-drill", x, 4, NORTH)

    drills = sum(1 for e in canvas.entities if e.name == "electric-mining-drill")
    blueprint = canvas.build(f"{drills} drills")
    return _oriented(blueprint, rng, data, "mining-outpost", None)


def lab_block(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A lab row with pass-through inserters and a science belt above.

    Labs hand science along the row, so the inserters between them face east —
    each takes from the lab behind it and inserts into the lab ahead.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    inserter = INSERTERS[min(tier + 1, len(INSERTERS) - 1)]
    count = rng.randint(2, 8)
    pitch = 4
    width = count * pitch - 1

    canvas.line(BELTS[tier], 0, 0, width, EAST)
    for index in range(count):
        x = index * pitch
        canvas.place("lab", x, 2)
        canvas.place(inserter, x, 1, SOUTH)
        if index:
            canvas.place(inserter, x - 1, 3, EAST)
    # One pole per pair of labs, in the free column between them, and never past
    # the last lab: a pole hanging off the end would stretch the bounding box
    # beyond the belt, leaving the belt's head pointing at an empty tile that is
    # suddenly *inside* the design.
    for gap in sorted({min(index, count - 2) for index in range(0, count, 2)}):
        canvas.maybe("medium-electric-pole", gap * pitch + 3, 1)

    blueprint = canvas.build(f"{count} labs")
    return _oriented(blueprint, rng, data, "lab-block", None)


def solar_block(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """Solar panels with an accumulator strip, in roughly the vanilla ratio.

    The exact ratio is 23.8 panels to 20.8 accumulators; a block of panel
    columns with one accumulator column per three panel columns lands near
    enough, and the point of the example is the tiling, not the power balance.
    """
    canvas = Canvas.new(data)
    columns = rng.randint(2, 6)
    rows = rng.randint(2, 5)
    x = 0
    panels = 0
    for column in range(columns):
        if column and column % 3 == 0:
            for row in range(rows * 3 // 2):
                canvas.place("accumulator", x, row * 2)
            x += 2
            continue
        for row in range(rows):
            canvas.place("solar-panel", x, row * 3)
            panels += 1
        x += 3
    for row in range(0, rows * 3, 6):
        canvas.maybe("medium-electric-pole", x, row)

    blueprint = canvas.build(f"{panels} panels")
    return _oriented(blueprint, rng, data, "solar-block", None)


def balancer(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A 2-to-2 or 4-to-4 balancer, built from splitter stages.

    Splitters are 2x1 facing north/south and 1x2 facing east/west; the belts
    between stages make the wiring legible to a model that only sees tiles.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt, split = BELTS[tier], SPLITTERS[tier]
    lanes = rng.choice((2, 4))
    stages = 1 if lanes == 2 else rng.choice((2, 3))
    run = 2

    x = 0
    for lane in range(lanes):
        canvas.line(belt, x, lane, run, EAST)
    x += run
    for stage in range(stages):
        # Alternating the offset is what makes a balancer balance: stage 0 pairs
        # lanes (0,1) and (2,3), stage 1 pairs (1,2), and every lane reaches
        # every other after enough stages.
        offset = stage % 2
        lane = offset
        while lane + 1 < lanes:
            canvas.place(split, x, lane, EAST)
            lane += 2
        for lane in range(lanes):
            if canvas.free(belt, x, lane, EAST):
                canvas.place(belt, x, lane, EAST)
        x += 1
        for lane in range(lanes):
            canvas.line(belt, x, lane, run, EAST)
        x += run

    blueprint = canvas.build(f"{lanes}-{lanes} balancer")
    return _oriented(blueprint, rng, data, "balancer", None)


def oil_block(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """Chemical plants in a row under a pipe header, unloading onto a belt.

    Fluids make this the one template where pipes carry the structure, and it is
    worth having in the corpus for exactly that reason: without it the model
    never sees a pipe run.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    machine = "chemical-plant"
    craftable = [
        name
        for name in CHEMICALS
        if name in data.recipes
        and data.recipes[name].category in data.entities[machine].crafting_categories
    ]
    recipe = rng.choice(craftable or ["plastic-bar"])
    count = rng.randint(2, 7)
    width = count * 3

    for x in range(width):
        canvas.place("pipe", x, 0)
    for index in range(count):
        canvas.place(machine, index * 3, 1, recipe=recipe)
    for index in range(count):
        canvas.place(INSERTERS[min(tier + 1, len(INSERTERS) - 1)], index * 3, 4, SOUTH)
    canvas.line(BELTS[tier], 0, 5, width, EAST)
    for x in range(1, width, 6):
        canvas.maybe("medium-electric-pole", x, 4)

    blueprint = canvas.build(f"{count}x {recipe}")
    return _oriented(blueprint, rng, data, "oil-block", recipe)


def bus_tap(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A main bus with a splitter tap and an underground crossing.

    This is the shape that ties the rest together: several parallel lanes, one
    of them split off, the rest carried under the branch. It is also the layout
    where a model most obviously either has or has not learned lane alignment.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt, under, split = BELTS[tier], UNDERGROUNDS[tier], SPLITTERS[tier]
    lanes = rng.randint(2, 6)
    # A lane is one entity per tile, so the bus is capped by area rather than by
    # length: eighty belt tiles is what fits in the context alongside the tap.
    length = rng.randint(10, max(10, 80 // lanes))
    tap = rng.randrange(3, max(4, length - 4))

    # The tap hangs off the bottom lane, so the splitter's second output lands on
    # a fresh row instead of stealing a bus lane.
    canvas.place(split, tap, lanes - 1, EAST)
    for lane in range(lanes):
        for x in range(length):
            if canvas.free(belt, x, lane, EAST):
                canvas.place(belt, x, lane, EAST)

    # One lane ducks under the others, which is the crossing every bus has.
    hop = min(4, int(data.entities[under].max_distance or 5))
    duck = rng.randrange(lanes)
    start = min(tap + 3, length - hop - 2)
    if start > 0:
        for x in range(start, start + hop + 1):
            canvas.entities = [e for e in canvas.entities if not (e.x == x and e.y == duck)]
            canvas.taken.pop((x, duck), None)
        canvas.place(under, start, duck, EAST, flow="input")
        canvas.place(under, start + hop, duck, EAST, flow="output")

    canvas.line(belt, tap + 1, lanes, 2, EAST)
    canvas.line(belt, tap + 3, lanes, 4, SOUTH)

    blueprint = canvas.build(f"{lanes}-lane bus tap")
    return _oriented(blueprint, rng, data, "bus-tap", None)


def _oriented(
    blueprint: Blueprint,
    rng: random.Random,
    data: Data,
    kind: str,
    product: str | None = None,
) -> tuple[Blueprint, Spec]:
    """Rotate to a random quarter turn and derive the spec from the result.

    Rotating inside the generator rather than in the augmentation pass means the
    spec's width and height describe the blueprint as written, and measuring the
    count with the same function the validator uses means a generator cannot
    promise something the grader would then mark as broken.
    """
    turned = rotate(blueprint, rng.randrange(4), data)
    return turned, Spec.measure(turned, kind, product, data)


GENERATORS: dict[str, Generator] = {
    "belt-lane": belt_lane,
    "smelter-column": smelter_column,
    "assembler-row": assembler_row,
    "mall-cell": mall_cell,
    "mining-outpost": mining_outpost,
    "lab-block": lab_block,
    "solar-block": solar_block,
    "balancer": balancer,
    "oil-block": oil_block,
    "bus-tap": bus_tap,
}

# How often each template is drawn. The weights are not uniform: the layouts a
# player would call "a factory" — smelters, assembler rows, malls — are worth
# more of the corpus than belt lanes, which the model masters in a few hundred
# steps and then keeps being taught.
WEIGHTS = {
    "belt-lane": 6,
    "smelter-column": 14,
    "assembler-row": 16,
    "mall-cell": 12,
    "mining-outpost": 10,
    "lab-block": 8,
    "solar-block": 6,
    "balancer": 6,
    "oil-block": 8,
    "bus-tap": 8,
}


def sample(rng: random.Random, data: Data | None = None) -> tuple[Blueprint, Spec]:
    """Draw one blueprint from the weighted mixture of templates."""
    data = data or load()
    names = list(GENERATORS)
    kind = rng.choices(names, weights=[WEIGHTS[name] for name in names])[0]
    return GENERATORS[kind](rng, data)
