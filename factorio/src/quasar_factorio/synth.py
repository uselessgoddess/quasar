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
from dataclasses import dataclass, replace

from . import plan as planner
from .augment import oriented
from .blueprint import Blueprint, Placement, rotate
from .grammar import Port, Spec
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
    # Plates onto a belt, or into a chest per furnace. The chest version is what
    # a starter base looks like before the bus exists, and it is the only thing
    # that varies the output row, so it doubles what this template can say.
    output = rng.choice(("belt", *CHESTS))

    canvas.line(belt, 0, 0, width, EAST)
    if output == "belt":
        canvas.line(belt, 0, out_row + 1, width, EAST)
    for index in range(count):
        x = index * pitch
        canvas.place(furnace, x, 2)
        canvas.place(inserter, x, 1, SOUTH)
        canvas.place(inserter, x, out_row, SOUTH)
        if output != "belt":
            canvas.place(output, x, out_row + 1)
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
    output = rng.choice(("belt", "belt", *CHESTS))

    canvas.line(belt, 0, 0, width, EAST)
    if output == "belt":
        canvas.line(belt, 0, 6, width, EAST)
    for index in range(count):
        x = index * 3
        canvas.place(machine, x, 2, recipe=recipe)
        canvas.place(inserter, x, 1, SOUTH)
        canvas.place(inserter, x, 5, SOUTH)
        if output != "belt":
            canvas.place(output, x, 6)
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
    # Fed by a belt or from a chest, unloading onto a belt or into a chest: all
    # four combinations are cells people actually build, and the cell is small
    # enough that the feed is most of what there is to vary.
    feed = rng.choice(("belt", *CHESTS[: tier + 2]))
    output = rng.choice(("belt", *CHESTS[: tier + 2]))

    if feed == "belt":
        canvas.line(BELTS[tier], 0, 0, 3, EAST)
    else:
        canvas.place(feed, 0, 0)
    canvas.place(inserter, 0, 1, SOUTH)
    canvas.place(machine, 0, 2, recipe=recipe)
    canvas.place(inserter, 0, 5, SOUTH)
    if output == "belt":
        canvas.line(BELTS[tier], 0, 6, 3, EAST)
    else:
        canvas.place(output, 0, 6)
    canvas.maybe("medium-electric-pole", 2, 1)

    blueprint = canvas.build(recipe)
    return _oriented(blueprint, rng, data, "mall-cell", recipe)


def mining_outpost(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """Rows of drills unloading onto a shared belt.

    Drills drop onto the tile they face, so a two-row outpost points both rows
    at the belt between them. The pitch is the drill's own width: an electric
    drill is 3x3 and a burner one 2x2, and the burner variant is a real
    early-game outpost rather than a downgrade — it needs no poles at all,
    which is worth having in a corpus where every other template is wired.

    For the electric drill, every third column trades its upper drill for a
    medium pole: nine tiles apart is exactly the pole's wire reach, and its 7x7
    supply from that slot still touches the drill row on the far side of the
    belt, so one pole line powers both rows.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt = BELTS[tier]
    drill = rng.choice(("electric-mining-drill", "electric-mining-drill", "burner-mining-drill"))
    burner = drill == "burner-mining-drill"
    pitch = data.entities[drill].width
    rows = rng.choice((1, 2, 2))
    # A multiple of three where poles are involved: the pole slot is every third
    # column, so any other count leaves a tail of drills past the last pole with
    # nothing to power them. A burner outpost has no such constraint. The
    # divisor is the entity budget — one belt tile per column of pitch, plus a
    # drill per row — kept under `COUNT_MAX` with room for the poles.
    step = 1 if burner else 3
    columns = rng.randint(1, 60 // (pitch + rows) // step) * step
    width = columns * pitch

    canvas.line(belt, 0, pitch if rows == 2 else 0, width, EAST)
    for index in range(columns):
        x = index * pitch
        # The pole replaces a drill rather than joining one: the row is solid,
        # so there is no spare tile to put it in.
        pole = not burner and index % 3 == 1
        if rows == 1:
            if pole:
                canvas.place("medium-electric-pole", x + 1, 2)
            else:
                canvas.place(drill, x, 1, NORTH)
            continue
        if pole:
            canvas.place("medium-electric-pole", x + 1, 1)
        else:
            canvas.place(drill, x, 0, SOUTH)
        canvas.place(drill, x, pitch + 1, NORTH)

    drills = sum(1 for entity in canvas.entities if entity.name == drill)
    blueprint = canvas.build(f"{drills} drills")
    return _oriented(blueprint, rng, data, "mining-outpost", None)


#: Labs fed from a belt, or from a chest per lab. Both are things people build:
#: the belt is the early-game answer and the chest the logistic-network one.
LAB_FEEDS = ("belt", "logistic-chest-requester", "logistic-chest-buffer", "steel-chest")

#: Longest lab row each shape affords before the entity count runs out. A belt
#: costs one entity per tile, so belt-fed rows are much shorter than chest-fed
#: ones, and a second row of labs roughly halves both.
LAB_LIMITS = {(True, 1): 8, (True, 2): 5, (False, 1): 11, (False, 2): 6}


def lab_block(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """One or two lab rows with pass-through inserters and science above.

    Labs hand science along the row, so the inserters between them face east —
    each takes from the lab behind it and inserts into the lab ahead. Where
    there are two rows, the inserters between them face south for the same
    reason: the second row is fed by the first, not by a second belt.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    inserter = INSERTERS[min(tier + 1, len(INSERTERS) - 1)]
    feed = rng.choice(LAB_FEEDS)
    rows = rng.choice((1, 1, 2))
    count = rng.randint(2, LAB_LIMITS[feed == "belt", rows])
    pitch = 4
    width = count * pitch - 1

    if feed == "belt":
        canvas.line(BELTS[tier], 0, 0, width, EAST)
    for index in range(count):
        x = index * pitch
        if feed != "belt":
            canvas.place(feed, x, 0)
        canvas.place(inserter, x, 1, SOUTH)
        for row in range(rows):
            canvas.place("lab", x, 2 + row * 4)
            if index:
                canvas.place(inserter, x - 1, 3 + row * 4, EAST)
            if row:
                canvas.place(inserter, x, 1 + row * 4, SOUTH)
    # One pole per pair of labs, in the free column between them, and never past
    # the last lab: a pole hanging off the end would stretch the bounding box
    # beyond the belt, leaving the belt's head pointing at an empty tile that is
    # suddenly *inside* the design. With two rows the pole line moves down to
    # the inserter row between them, where one 7x7 supply covers both.
    for gap in sorted({min(index, count - 2) for index in range(0, count, 2)}):
        canvas.maybe("medium-electric-pole", gap * pitch + 3, 1 if rows == 1 else 5)

    blueprint = canvas.build(f"{count * rows} labs")
    return _oriented(blueprint, rng, data, "lab-block", None)


def solar_block(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """Solar panels with an accumulator strip, in roughly the vanilla ratio.

    The exact ratio is 23.8 panels to 20.8 accumulators; a block of panel
    columns with one accumulator column per three or four panel columns lands
    near enough, and the point of the example is the tiling, not the power
    balance.

    This is the one template where the pole choice is free rather than
    load-bearing — nothing here consumes power — so it draws from all three,
    which is also what real solar fields look like.
    """
    canvas = Canvas.new(data)
    pole = rng.choice(("medium-electric-pole", "big-electric-pole", "substation"))
    ratio = rng.choice((3, 4))
    columns = rng.randint(2, 8)
    rows = rng.randint(2, 8)
    # A panel column is `rows` entities and an accumulator column half again as
    # many, so a wide field has to be short to stay inside `COUNT_MAX`.
    while columns * rows * 5 // 4 > 56:
        rows -= 1

    x = 0
    panels = 0
    for column in range(columns):
        if column and column % ratio == 0:
            for row in range(rows * 3 // 2):
                canvas.place("accumulator", x, row * 2)
            x += 2
            continue
        for row in range(rows):
            canvas.place("solar-panel", x, row * 3)
            panels += 1
        x += 3
    for row in range(0, rows * 3, 6):
        canvas.maybe(pole, x, row)

    blueprint = canvas.build(f"{panels} panels")
    return _oriented(blueprint, rng, data, "solar-block", None)


def balancer(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """An n-to-n balancer, built from splitter stages.

    Splitters are 2x1 facing north/south and 1x2 facing east/west; the belts
    between stages make the wiring legible to a model that only sees tiles.

    Lane counts, stage counts and the belt runs between stages all vary, which
    is the only reason this template contributes more than a handful of designs
    to the corpus: rotating a balancer produces a blueprint the corpus builder
    already has, so every distinct example has to come from geometry.
    """
    canvas = Canvas.new(data)
    tier = _tier(rng)
    belt, split = BELTS[tier], SPLITTERS[tier]
    lanes = rng.choice((2, 2, 3, 4, 4, 6, 8))
    stages = rng.randint(1, 3)
    lead, run = rng.randint(1, 4), rng.randint(1, 4)
    # Every tile of every lane is one entity, so a wide balancer has to be
    # short. Shrinking the runs rather than the lane count keeps the shape the
    # draw asked for; `#N` tops out at COUNT_MAX and the grader is unforgiving.
    while lanes * (lead + stages * (1 + run)) > 60:
        if run > 1:
            run -= 1
        elif lead > 1:
            lead -= 1
        else:
            stages -= 1

    x = 0
    for lane in range(lanes):
        canvas.line(belt, x, lane, lead, EAST)
    x += lead
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
    buried = rng.random() < 0.5
    # A tank on the head of the header is what the fluid actually arrives from,
    # and it is the only 3x3 non-crafting building in the corpus.
    left = 4 if rng.random() < 0.5 else 0

    if left:
        canvas.place("storage-tank", 0, 0)
    for x in range(left, left + width):
        # Every third tile is where a plant's fluid box meets the header, so
        # those stay above ground and the spans between them duck under —
        # which is how a header is built once undergrounds are affordable.
        if not buried or (x - left) % 3 == 1:
            canvas.place("pipe", x, 0)
        elif (x - left) % 3 == 2:
            canvas.place("pipe-to-ground", x, 0, EAST)
        elif x - left:
            canvas.place("pipe-to-ground", x, 0, WEST)
    if left:
        canvas.place("pipe", left - 1, 0)
    for index in range(count):
        x = left + index * 3
        canvas.place(machine, x, 1, recipe=recipe)
        canvas.place(INSERTERS[min(tier + 1, len(INSERTERS) - 1)], x, 4, SOUTH)
    canvas.line(BELTS[tier], left, 5, width, EAST)
    for x in range(left + 1, left + width, 6):
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


#: Entities one module may contain. A document is roughly six tokens an entity
#: and the context is 512, so this is the ceiling that keeps a module — spec,
#: ports, plan and all — inside one window rather than truncated at the end,
#: which would teach the model that blueprints sometimes just stop.
MODULE_ENTITIES = 70

#: Modules are wired with medium poles and no choice about it. A small pole
#: covers 5x5, and the bottom row of inserters of the last stage sits four tiles
#: below the row of poles that has to reach it — exactly the medium pole's
#: limit, and one tile past the small one's. Offering a tier that silently
#: unpowers the output row of every assembler module is not variety.
MODULE_POLE = "medium-electric-pole"


@functools.cache
def _module_targets(data: Data) -> tuple[planner.Module, ...]:
    """Every product a stacked-band module can make, with what it is handed."""
    return planner.modules(data)


@dataclass(frozen=True)
class Layout:
    """How a module is spaced out, as opposed to what it builds.

    Without these the generator was a lookup table: the catalogue has twenty
    entries and the machine budget only two or three settings, so it saturated
    at forty-five distinct designs and every draw after the first few hundred
    was a duplicate the deduplicator threw away. Measured by
    `experiments/module_yield.py`.

    A knob only counts if `augment.canonical` cannot see through it, which rules
    out the obvious two: it downgrades every belt, inserter and machine to the
    bottom of its upgrade chain before comparing, so tier randomisation adds
    nothing, and it takes the minimum over the dihedral group, so rotating and
    mirroring add nothing either. What survives is where things sit — spacing,
    alignment, pole cadence, and which way along the row each belt runs.
    """

    #: Spare columns between neighbouring machines in a stage.
    gap: int
    #: Columns of belt beyond the widest stage, so the zone is not skin-tight.
    margin: int
    #: How far apart poles go along a row of inserters.
    cadence: int
    #: Where a stage narrower than the widest one sits: `left`, `centre`, `right`.
    align: str

    @classmethod
    def draw(cls, rng: random.Random) -> Layout:
        """Biased toward the tight packing, because that is what players build."""
        return cls(
            gap=rng.choices((0, 1, 2), weights=(5, 3, 2))[0],
            margin=rng.choices((0, 1, 2), weights=(5, 3, 2))[0],
            cadence=rng.randint(4, 6),
            align=rng.choice(("left", "centre", "right")),
        )

    def offset(self, span: int, width: int) -> int:
        """Where a stage of `span` columns starts inside a `width`-wide module."""
        pad = width - span
        return {"left": 0, "centre": pad // 2, "right": pad}[self.align]


def module(rng: random.Random, data: Data) -> tuple[Blueprint, Spec]:
    """A closed module: inputs on the edge, a recipe chain inside, one output.

    This is the layout the whole planner exists for, and it is one idea repeated
    once per stage of the chain. A stage is a belt, a row of inserters facing
    down into a row of machines, and a row of inserters facing down out of them —
    so the belt *below* one stage is the belt *above* the next, and the product
    of the machines above lands on exactly the belt the machines below draw from.
    A two-stage chain stacks two of those and the intermediate never touches the
    module boundary at all.

    Raw ingredients enter at the upstream end of whichever stage needs them,
    which is why `plan.modules` rejects a chain whose stage wants more raws than
    a belt has lanes. The finished product leaves at the far end of the bottom
    belt, and that tile is the only place anything crosses the edge: every other
    belt row stops one column short and ends against a pole, so nothing spills
    into whatever the player puts next to this.

    Everything the layout is free to choose lives in :class:`Layout`, plus the
    direction each belt row runs, which is drawn per row rather than per design:
    flipping every row together is a mirror image and the augmenter already
    produces those, while flipping one row is a design the corpus has not seen.
    Which flips are available is geometry and not taste — see `_runs`.
    """
    target = rng.choice(_module_targets(data))
    product, supply = target.product, target.supply
    unit = planner.solve(data, product, supply, rate=1e-9, depth=target.depth)
    layout = Layout.draw(rng)
    stages = _budgeted(data, target, unit, rng, layout)

    tier = _tier(rng)
    belt = BELTS[tier]
    inserter = INSERTERS[min(tier + 1, len(INSERTERS) - 1)]
    canvas = Canvas.new(data)
    width = _module_width(data, stages, layout)
    columns = _columns(data, stages, layout, width)
    runs = _runs(rng, columns, width)

    ports: list[Port] = []
    row = 0
    for index, stage in enumerate(stages):
        proto = data.entities[stage.machine]
        # One column short of the edge: the belt is internal plumbing, and a
        # belt that reached the boundary would be an undeclared port.
        head, _, side = _belt_row(canvas, belt, row, width, runs[index], full=False)
        for x in columns[index]:
            canvas.place(inserter, x, row + 1, SOUTH)
            canvas.place(stage.machine, x, row + 2, recipe=stage.recipe)
            canvas.place(inserter, x, row + 2 + proto.height, SOUTH)
        _power_row(canvas, row + 1, width, layout.cadence)
        for item in dict.fromkeys(stage.ingredients):
            if item in supply:
                ports.append(Port("in", item, head, row, side))
        row += 3 + proto.height

    _, tail, side = _belt_row(canvas, belt, row, width, runs[-1], full=True)
    ports.append(Port("out", product, tail, row, "e" if side == "w" else "w"))

    blueprint = canvas.build(f"{product} module")
    steps = tuple(stage.step() for stage in stages)
    # A zone the player marked out is not always the zone a design fills, so
    # some of the corpus is drawn a little smaller than it was asked for. The
    # spec then states the request and the grader checks the design fits inside
    # it, which is the relation that holds at inference time.
    slack = rng.choices((0, 1, 2), weights=(6, 2, 1))[0]
    return _module_spec(blueprint, ports, steps, product, slack, rng, data)


def _belt_row(
    canvas: Canvas, belt: str, row: int, width: int, direction: int, full: bool
) -> tuple[int, int, str]:
    """One belt row of a module, and the two ends items travel between.

    Returns the upstream column, the downstream column and the side the
    upstream end is on, because that is what a port needs: an item put on the
    downstream end of a belt has already left, so an input port belongs at the
    head and an output port at the tail.

    A `full` row spans the whole width and its tail points off the edge, which
    is the module's one deliberate opening. Every other row stops a column short
    and butts against a pole, so the belt leads somewhere without leaking.
    """
    length = width if full else width - 1
    step = 1 if direction == EAST else -1
    head = 0 if direction == EAST else width - 1
    canvas.line(belt, head, row, length, direction)
    if not full:
        canvas.maybe(MODULE_POLE, head + step * length, row)
    return head, head + step * (length - 1), "w" if direction == EAST else "e"


def _power_row(canvas: Canvas, row: int, width: int, cadence: int) -> None:
    """Poles along a row of inserters, at the first free column each cadence.

    Stepping blindly and giving up when the tile is taken leaves holes: machines
    packed edge to edge put an inserter on every third column, so a cadence of
    six lands on one every other time and the machines past it lose power.
    Sliding to the next free column keeps the run unbroken, and since a run of
    occupied columns is never longer than a machine is wide, it never slides far
    enough to open a gap the next pole cannot cover.
    """
    last = -cadence
    for x in range(width):
        if x - last >= cadence and canvas.maybe(MODULE_POLE, x, row):
            last = x


def _columns(data: Data, stages, layout: Layout, width: int) -> list[list[int]]:
    """Which columns each stage's machines stand in.

    Alignment is a per-design choice, so a stage narrower than the widest one is
    the only stage it moves, and moving it is what makes two draws of the same
    plan two different designs. Centring is given up on when it would strand a
    row — see `_directions` — because a layout that cannot be belted is not a
    variant of this design, it is a broken one.
    """
    columns = _placed(data, stages, layout, width)
    if all(_directions(columns, row, width) for row in range(len(columns) + 1)):
        return columns
    return _placed(data, stages, replace(layout, align="left"), width)


def _placed(data: Data, stages, layout: Layout, width: int) -> list[list[int]]:
    columns = []
    for stage in stages:
        proto = data.entities[stage.machine]
        span = stage.count * proto.width + (stage.count - 1) * layout.gap
        offset = layout.offset(span, width)
        stride = proto.width + layout.gap
        columns.append([offset + index * stride for index in range(stage.count)])
    return columns


def _directions(columns: list[list[int]], row: int, width: int) -> tuple[int, ...]:
    """Which ways belt row `row` may flow, on the two counts that decide it.

    *Reach.* A belt hands items downstream and nowhere else, so every machine
    dropping onto a row has to have one upstream of every machine picking up
    from it: eastward needs the leftmost drop no further right than the leftmost
    pickup, westward the mirror image. The top row has no drops and the bottom
    row no pickups, so neither is constrained this way.

    *The end pole.* An internal row stops a column short and butts against a
    pole, and the column it stops at is the downstream one — column 0 for a
    westward row. An inserter standing over that column would reach past the
    belt entirely and try to take items out of the pole, so a row whose end
    lands under a machine cannot run that way. Aligned left, that rules out
    westward for every internal row, which is why the margin is worth drawing:
    one spare column at the edge is what buys the flip.

    Centring a narrow stage over a wide one is the case that can leave neither
    direction working, with the drops strictly inside the span of the pickups.
    Aligning both stages to the same edge always leaves eastward, which is why
    `_columns` falls back there rather than emitting a module that starves.
    """
    drops = columns[row - 1] if row else []
    pickups = columns[row] if row < len(columns) else []
    allowed = []
    for direction in (EAST, WEST):
        end = width - 1 if direction == EAST else 0
        if row < len(columns) and end in set(drops) | set(pickups):
            continue
        if drops and pickups:
            reaches = (
                min(drops) <= min(pickups) if direction == EAST else max(drops) >= max(pickups)
            )
            if not reaches:
                continue
        allowed.append(direction)
    return tuple(allowed)


def _runs(rng: random.Random, columns: list[list[int]], width: int) -> list[int]:
    """A flow direction per belt row, drawn from the ones that reach."""
    return [rng.choice(_directions(columns, row, width)) for row in range(len(columns) + 1)]


def _module_width(data: Data, stages, layout: Layout) -> int:
    """Wide enough for the widest stage, plus whatever margin was asked for."""
    return layout.margin + max(
        stage.count * data.entities[stage.machine].width + (stage.count - 1) * layout.gap
        for stage in stages
    )


def _budgeted(data: Data, target: planner.Module, unit, rng: random.Random, layout: Layout):
    """The largest plan whose layout still fits `MODULE_ENTITIES`.

    Searched downwards from a random ambition rather than solved: the entity
    count is a step function of the machine count, because a wider stage extends
    every belt row in the design and not only its own.
    """
    for machines in range(rng.randint(unit.machines, unit.machines * 3), unit.machines - 1, -1):
        try:
            candidate = planner.fit(
                data, target.product, target.supply, machines=machines, depth=target.depth
            )
        except planner.PlanError:
            continue
        if _entities(data, candidate.stages, layout) <= MODULE_ENTITIES:
            return candidate.stages
    return unit.stages


def _entities(data: Data, stages, layout: Layout) -> int:
    """What `module` will place, without placing it.

    An upper bound rather than a count: `_power_row` slides its poles past
    occupied columns and may fit one fewer than the cadence suggests. Over-
    estimating costs a machine off the plan; under-estimating costs a truncated
    document, which teaches the model that a blueprint can simply stop.
    """
    width = _module_width(data, stages, layout)
    poles = -(-width // layout.cadence)
    total = width  # the output belt, which runs the full width
    for stage in stages:
        # belt, end pole, three entities a machine, and the poles over them
        total += (width - 1) + 1 + 3 * stage.count + poles
    return total


def _module_spec(
    blueprint: Blueprint,
    ports: list[Port],
    steps: tuple,
    product: str,
    slack: int,
    rng: random.Random,
    data: Data,
) -> tuple[Blueprint, Spec]:
    """Rotate the module and its ports together, then state the zone."""
    turned, moved = oriented(blueprint, tuple(ports), rng.randrange(4), data=data)
    width, height = turned.extent(data)
    return turned, Spec.measure(
        turned,
        "module",
        product,
        data,
        ports=moved,
        plan=steps,
        zone=(width + slack, height + slack),
    )


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
    "module": module,
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
#
# `module` is the heaviest by a wide margin because it is the only task in the
# mixture that is not already solved. The measured run put every other metric
# above 0.97 and then had nothing left to learn from those templates; a quarter
# of the corpus being the unsolved task is what makes the next run's loss curve
# mean something.
WEIGHTS = {
    "module": 26,
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
