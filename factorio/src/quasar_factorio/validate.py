"""Does this thing actually paste into Factorio, and is it any good?

Perplexity is the wrong headline metric for this task. A model can shave a
quarter nat off the loss and still put two assemblers on the same tile, which
the game rejects outright with "entities are overlapping". So the harness grades
generations the way the game does, and separately scores the things a player
would notice.

Two tiers:

*Validity* is binary and unforgiving — parses, fits the grid, no overlaps, every
recipe legal for the machine it is in. This is the number the issue asks for and
the one plotted as the headline.

*Quality* is continuous: powered fraction, inserters with something on both
sides, belts that lead somewhere. A blueprint can be perfectly valid and
completely useless, and without these three the validity curve would happily
report success for a field of disconnected furnaces.

*Flow* is the third tier and the newest, delegated to `flow.trace`. The first
two tiers are local — each asks a question about one entity and its neighbours —
and the measured run saturated all of them at once, which is what a benchmark
does when it has stopped measuring the model rather than when the model has
finished learning. Whether an item can travel from an input to an output is not
a question any single tile can answer, and it is the question a module exists to
get right.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import flow as itemflow
from .blueprint import GRID, Blueprint
from .grammar import COUNT_MAX, ParseError, Spec, parse
from .prototypes import EAST, NORTH, SOUTH, WEST, Data, load

# Where an inserter of each direction takes from and puts to. A Factorio
# inserter faces the tile it *inserts into*, and picks up from the opposite
# side, which is the single most common thing to get backwards.
_OFFSETS = {
    NORTH: ((0, 1), (0, -1)),
    EAST: ((-1, 0), (1, 0)),
    SOUTH: ((0, -1), (0, 1)),
    WEST: ((1, 0), (-1, 0)),
}

# Belt output direction, in tiles.
_AHEAD = {NORTH: (0, -1), EAST: (1, 0), SOUTH: (0, 1), WEST: (-1, 0)}

# Categories that stop working without electricity. Burner furnaces and burner
# inserters are deliberately absent: they are powered by coal and counting them
# as unpowered would slander every early-game smelter design in the corpus.
_NEEDS_POWER = {
    "assembling-machine",
    "inserter",
    "lab",
    "beacon",
    "mining-drill",
    "roboport",
    "radar",
    "pump",
    "rocket-silo",
}
_BURNER = {"stone-furnace", "steel-furnace", "burner-inserter", "burner-mining-drill"}

# Things an inserter can meaningfully touch.
_TRANSFERABLE = {
    "assembling-machine",
    "furnace",
    "transport-belt",
    "underground-belt",
    "splitter",
    "container",
    "logistic-container",
    "lab",
    "boiler",
    "rocket-silo",
    "inserter",
    "mining-drill",
}


@dataclass
class Report:
    """One blueprint, graded."""

    parsed: bool = False
    entities: int = 0
    tokens: int = 0
    width: int = 0
    height: int = 0
    fits: bool = False
    overlaps: int = 0
    illegal_recipes: int = 0
    valid: bool = False
    powered: float = 0.0
    connected_inserters: float = 0.0
    belts_lead_somewhere: float = 0.0
    error: str = ""
    error_at: int = -1
    spec_kind: str = ""
    spec_honoured: bool = False
    #: Whether the spec declared output ports, i.e. whether `delivers` means
    #: anything. A belt lane has nothing to deliver and should not be averaged
    #: in as a failure to deliver it.
    ported: bool = False
    delivers: float = 0.0
    fed: float = 0.0
    working: float = 0.0
    #: Crafting machines in the design; see `flow.Flow.machines`. The same
    #: caveat `ported` carries for `delivers`, carried for `fed` and `working`.
    machines: int = 0
    mixed: int = 0
    leaks: int = 0
    within_zone: bool = True
    missing: tuple[str, ...] = ()

    def asks_about_flow(self) -> bool:
        """Whether `flows()` is a fact about this design or an artefact.

        A balancer has no ports and no machines: nothing was supposed to arrive
        anywhere, so there is no sense in which it failed to. It is a legality
        question, it scores 1.000 on the legality tier, and averaging a flow of
        zero in for it would make the flow number report the composition of the
        benchmark rather than the state of the model.
        """
        return self.valid and (self.ported or bool(self.machines))

    def quality(self) -> float:
        """One number for a scatter plot, not for a leaderboard.

        Zero for anything invalid, otherwise the mean of the three quality
        fractions. Deliberately blunt: its job is to separate "a field of
        disconnected furnaces" from "a smelter".

        Flow is deliberately *not* folded in. The two are measuring different
        failures — decoration versus a broken chain — and a single average would
        let one hide the other, which is how the run being analysed ended up
        with a headline number that had nothing left to say.
        """
        if not self.valid:
            return 0.0
        return (self.powered + self.connected_inserters + self.belts_lead_somewhere) / 3

    def flows(self) -> float:
        """The item-flow score: does the thing work, as opposed to parse?

        Zero for anything invalid, for the same reason `quality` is: an
        overlapping design cannot be pasted, so what would have flowed through
        it is not a fact about anything.

        Also zero for a design with neither ports nor machines, which is a
        different kind of nothing — `asks_about_flow` is the one to ask before
        averaging this, and `summarise` does.
        """
        if not self.valid:
            return 0.0
        return 0.7 * self.delivers + 0.3 * self.fed if self.ported else self.fed

    def to_dict(self) -> dict:
        out = asdict(self)
        out["missing"] = list(self.missing)
        return out


@dataclass
class Summary:
    """Aggregate over a batch of generations, for the metrics log."""

    samples: int = 0
    parse_rate: float = 0.0
    valid_rate: float = 0.0
    mean_entities: float = 0.0
    mean_quality: float = 0.0
    mean_powered: float = 0.0
    mean_connected: float = 0.0
    mean_belts: float = 0.0
    overlap_rate: float = 0.0
    empty_rate: float = 0.0
    spec_rate: float = 0.0
    #: `mean_delivers` is averaged over the samples that declared output ports;
    #: `mean_fed` and `mean_working` over the ones with a machine to feed;
    #: `mean_flow` over either. The three counts are here so a low number can be
    #: read as "few were asked" rather than mistaken for "few succeeded".
    ported: int = 0
    crafting: int = 0
    flowing: int = 0
    mean_delivers: float = 0.0
    mean_fed: float = 0.0
    mean_working: float = 0.0
    mean_flow: float = 0.0
    leak_rate: float = 0.0
    mixed_rate: float = 0.0
    zone_rate: float = 0.0
    errors: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def occupancy(blueprint: Blueprint, data: Data) -> dict[tuple[int, int], list[int]]:
    """Tile -> the indices of the entities claiming it."""
    grid: dict[tuple[int, int], list[int]] = {}
    for index, placement in enumerate(blueprint.entities):
        for tile in placement.tiles(data):
            grid.setdefault(tile, []).append(index)
    return grid


def overlapping(blueprint: Blueprint, data: Data) -> int:
    """Entities sharing a tile with another entity.

    Counted per entity rather than per tile: one assembler dropped on top of
    another is two offending entities, not nine offending tiles.
    """
    guilty = set()
    for holders in occupancy(blueprint, data).values():
        if len(holders) > 1:
            guilty.update(holders)
    return len(guilty)


def illegal_recipes(blueprint: Blueprint, data: Data) -> int:
    """Recipes the assigned machine cannot run.

    A chemical plant set to `iron-gear-wheel` is not an overlap error, but the
    game will not build it either, so it counts against validity.
    """
    bad = 0
    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        if proto is None or not placement.recipe:
            continue
        recipe = data.recipes.get(placement.recipe)
        if recipe is None or recipe.category not in proto.crafting_categories:
            bad += 1
    return bad


def powered_fraction(blueprint: Blueprint, data: Data) -> float:
    """Fraction of electric entities inside some pole's supply area.

    Approximated as the pole's square supply area around its centre, which is
    how the game draws it.
    """
    poles = []
    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        if proto is None or proto.category != "electric-pole":
            continue
        width, height = proto.footprint(placement.direction)
        centre = (placement.x + width / 2, placement.y + height / 2)
        poles.append((centre, proto.supply_area or 0.0))

    wanted, covered = 0, 0
    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        if proto is None or placement.name in _BURNER or proto.category not in _NEEDS_POWER:
            continue
        wanted += 1
        width, height = proto.footprint(placement.direction)
        centre = (placement.x + width / 2, placement.y + height / 2)
        if any(
            abs(centre[0] - px) <= area + width / 2 and abs(centre[1] - py) <= area + height / 2
            for (px, py), area in poles
        ):
            covered += 1
    return covered / wanted if wanted else 1.0


def connected_inserters(blueprint: Blueprint, data: Data) -> float:
    """Fraction of inserters with something to grab from *and* somewhere to put.

    This is the cheapest test that distinguishes a learned factory from learned
    decoration, and it is the metric that moves first during training.
    """
    grid = occupancy(blueprint, data)

    def transferable(tile) -> bool:
        for index in grid.get(tile, ()):
            proto = data.entity(blueprint.entities[index].name)
            if proto is not None and proto.category in _TRANSFERABLE:
                return True
        return False

    inserters = [
        placement
        for placement in blueprint.entities
        if (proto := data.entity(placement.name)) is not None and proto.category == "inserter"
    ]
    if not inserters:
        return 1.0

    good = 0
    for placement in inserters:
        offsets = _OFFSETS.get(placement.direction)
        if offsets is None:
            continue
        # From the prototype's own pickup position, not from the name: a mod or
        # a future version could add a third reach and this stays right.
        reach = int(data.entities[placement.name].max_distance or 1)
        (fx, fy), (tx, ty) = offsets
        source = (placement.x + fx * reach, placement.y + fy * reach)
        sink = (placement.x + tx * reach, placement.y + ty * reach)
        if transferable(source) and transferable(sink):
            good += 1
    return good / len(inserters)


def belts_lead_somewhere(blueprint: Blueprint, data: Data) -> float:
    """Fraction of belts whose forward tile is occupied or off the design.

    A belt pointing at empty space *inside* the blueprint is a mistake; a belt
    pointing off the edge is an output, which is the whole point of a belt lane,
    so those count as fine.
    """
    grid = occupancy(blueprint, data)
    x0, y0, x1, y1 = blueprint.bounds(data)
    belts = [
        placement
        for placement in blueprint.entities
        if (proto := data.entity(placement.name)) is not None
        and proto.category in {"transport-belt", "underground-belt"}
    ]
    if not belts:
        return 1.0

    def tunnel_lands(placement, step) -> bool:
        """Whether an underground entrance has a matching exit down the line.

        The tile in front of an entrance is *supposed* to be empty — that is the
        tunnel — so the forward-tile test slanders every correctly built pair.
        """
        proto = data.entity(placement.name)
        reach = int(proto.max_distance or 0)
        for distance in range(1, reach + 1):
            tile = (placement.x + step[0] * distance, placement.y + step[1] * distance)
            for index in grid.get(tile, ()):
                other = blueprint.entities[index]
                if other.name == placement.name and other.direction == placement.direction:
                    return other.flow == "output"
        return False

    good = 0
    for placement in belts:
        step = _AHEAD.get(placement.direction)
        if step is None:
            continue
        if placement.flow == "input":
            good += tunnel_lands(placement, step)
            continue
        ahead = (placement.x + step[0], placement.y + step[1])
        outside = not (x0 <= ahead[0] < x1 and y0 <= ahead[1] < y1)
        if outside or ahead in grid:
            good += 1
    return good / len(belts)


def inspect(blueprint: Blueprint, data: Data | None = None, spec: Spec | None = None) -> Report:
    """Grade an already-parsed blueprint."""
    data = data or load()
    width, height = blueprint.extent(data)
    report = Report(
        parsed=True,
        entities=len(blueprint.entities),
        width=width,
        height=height,
        fits=blueprint.fits(data, GRID),
        overlaps=overlapping(blueprint, data),
        illegal_recipes=illegal_recipes(blueprint, data),
        powered=powered_fraction(blueprint, data),
        connected_inserters=connected_inserters(blueprint, data),
        belts_lead_somewhere=belts_lead_somewhere(blueprint, data),
        spec_kind=spec.kind if spec else "",
    )
    report.valid = bool(
        report.entities and report.fits and report.overlaps == 0 and report.illegal_recipes == 0
    )
    if report.valid:
        # Only for a design the game would accept. Propagating items through a
        # pile of overlapping entities is neither meaningful nor cheap, and the
        # answer is discarded by `flows()` anyway.
        traced = itemflow.trace(blueprint, spec, data)
        report.ported = bool(spec is not None and spec.outputs())
        report.delivers = traced.delivers
        report.fed = traced.fed
        report.working = traced.working
        report.machines = traced.machines
        report.mixed = traced.mixed
        report.leaks = traced.leaks
        report.within_zone = traced.within_zone
        report.missing = traced.missing
    if spec is not None:
        # A tolerance of two machines and four tiles. Asking a 3M model to hit an
        # exact entity count is asking it to count, which is not what it is for.
        crafters = sum(
            1
            for placement in blueprint.entities
            if (proto := data.entity(placement.name)) is not None and proto.crafting_categories
        )
        asked = crafters or report.entities
        # `#64` is the largest count token there is, so it means "at least 64"
        # rather than "exactly 64"; holding a saturated spec to +/-2 would fail
        # every large belt lane for a limit of the vocabulary, not of the model.
        counted = asked >= COUNT_MAX if spec.count >= COUNT_MAX else abs(asked - spec.count) <= 2
        report.spec_honoured = bool(
            report.valid
            and counted
            and abs(width - spec.width) <= 4
            and abs(height - spec.height) <= 4
        )
    return report


def grade(text: str, data: Data | None = None) -> Report:
    """Grade a raw generation, parse failure included."""
    data = data or load()
    report = Report(tokens=len(text.split()))
    try:
        blueprint, spec = parse(text, data)
    except ParseError as error:
        report.error = str(error)
        report.error_at = error.at
        report.entities = error.parsed
        return report
    graded = inspect(blueprint, data, spec)
    graded.tokens = report.tokens
    return graded


def summarise(reports: list[Report]) -> Summary:
    """Aggregate, with the error histogram that says *why* things failed."""
    if not reports:
        return Summary()
    total = len(reports)
    ok = [report for report in reports if report.parsed]
    errors: dict[str, int] = {}
    for report in reports:
        if report.error:
            # Keep the shape of the message, not its payload, so that fifty
            # different unknown entities collapse into one bar.
            key = report.error.split(",")[0].split("'")[0].strip()
            errors[key] = errors.get(key, 0) + 1

    def mean(values) -> float:
        values = list(values)
        return sum(values) / len(values) if values else 0.0

    specced = [report for report in reports if report.spec_kind]
    # Flow is only defined for designs that were built; averaging a zero in for
    # every unparseable generation would make the metric track validity instead
    # of flow, and the two are supposed to be able to disagree.
    built = [report for report in reports if report.valid]
    ported = [report for report in built if report.ported]
    # And only for designs the question applies to. A belt lane has no machine
    # to feed, so the zero `fed` it reports is "nothing was asked" — averaging
    # those in makes the number track how much of the benchmark is machinery.
    crafting = [report for report in built if report.machines]
    flowing = [report for report in reports if report.asks_about_flow()]
    return Summary(
        samples=total,
        parse_rate=len(ok) / total,
        valid_rate=sum(report.valid for report in reports) / total,
        mean_entities=mean(report.entities for report in ok),
        mean_quality=mean(report.quality() for report in reports),
        mean_powered=mean(report.powered for report in ok),
        mean_connected=mean(report.connected_inserters for report in ok),
        mean_belts=mean(report.belts_lead_somewhere for report in ok),
        overlap_rate=mean(bool(report.overlaps) for report in ok),
        empty_rate=mean(report.entities == 0 for report in reports),
        spec_rate=mean(report.spec_honoured for report in specced) if specced else 0.0,
        ported=len(ported),
        crafting=len(crafting),
        flowing=len(flowing),
        mean_delivers=mean(report.delivers for report in ported),
        mean_fed=mean(report.fed for report in crafting),
        mean_working=mean(report.working for report in crafting),
        mean_flow=mean(report.flows() for report in flowing),
        leak_rate=mean(bool(report.leaks) for report in built),
        mixed_rate=mean(bool(report.mixed) for report in built),
        zone_rate=mean(report.within_zone for report in ported),
        errors=dict(sorted(errors.items(), key=lambda item: -item[1])),
    )
