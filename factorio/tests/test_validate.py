"""The grader. If this is wrong, every number in the plots is wrong."""

from quasar_factorio import grammar, prototypes, validate
from quasar_factorio.blueprint import Blueprint, Placement
from quasar_factorio.grammar import Port, Spec
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST

DATA = prototypes.load()


def graded(*placements, spec=None) -> validate.Report:
    blueprint = Blueprint(entities=list(placements)).normalised(DATA)
    return validate.inspect(blueprint, DATA, spec)


def test_two_entities_on_one_tile_are_both_guilty():
    report = graded(
        Placement("assembling-machine-1", 0, 0),
        Placement("assembling-machine-1", 1, 1),
    )
    assert report.overlaps == 2
    assert not report.valid


def test_entities_that_merely_touch_do_not_overlap():
    report = graded(
        Placement("assembling-machine-1", 0, 0),
        Placement("assembling-machine-1", 3, 0),
    )
    assert report.overlaps == 0
    assert report.valid


def test_a_rotated_splitter_occupies_the_rotated_tiles():
    # 2x1 facing north, 1x2 facing east: placing one of each at (0,0) and (0,1)
    # only fits because the eastward one is a tall column.
    report = graded(Placement("splitter", 0, 0, NORTH), Placement("splitter", 0, 1, EAST))
    assert report.overlaps == 0
    assert (report.width, report.height) == (2, 3)


def test_a_recipe_the_machine_cannot_run_is_invalid():
    report = graded(Placement("assembling-machine-1", 0, 0, recipe="sulfuric-acid"))
    assert report.illegal_recipes == 1
    assert not report.valid

    report = graded(Placement("chemical-plant", 0, 0, recipe="sulfuric-acid"))
    assert report.illegal_recipes == 0
    assert report.valid


def test_an_empty_blueprint_is_not_valid():
    assert not validate.inspect(Blueprint(), DATA).valid


def test_inserters_face_what_they_insert_into():
    # Belt above, furnace below, inserter between facing south: it picks up from
    # the belt behind it and puts into the furnace ahead.
    good = graded(
        Placement("transport-belt", 0, 0, EAST),
        Placement("inserter", 0, 1, SOUTH),
        Placement("stone-furnace", 0, 2),
    )
    assert good.connected_inserters == 1.0

    # The same layout with the inserter turned around grabs from the furnace and
    # inserts into the belt, which is also connected — but turned sideways it
    # reaches into empty space on both sides.
    sideways = graded(
        Placement("transport-belt", 0, 0, EAST),
        Placement("inserter", 0, 1, EAST),
        Placement("stone-furnace", 0, 2),
    )
    assert sideways.connected_inserters == 0.0


def test_a_long_handed_inserter_reaches_two_tiles():
    layout = (
        Placement("transport-belt", 0, 0, EAST),
        Placement("long-handed-inserter", 0, 2, NORTH),
        Placement("stone-furnace", 0, 3),
    )
    assert graded(*layout).connected_inserters == 1.0
    # A normal inserter in the same slot reaches nothing.
    short = list(layout)
    short[1] = Placement("inserter", 0, 2, NORTH)
    assert graded(*short).connected_inserters == 0.0


def test_power_coverage_uses_the_real_supply_area():
    # A medium pole supplies 7x7. An assembler four tiles to the right of it
    # still touches that square; one eight tiles away does not.
    near = graded(
        Placement("medium-electric-pole", 0, 0),
        Placement("assembling-machine-1", 4, 0, recipe="iron-gear-wheel"),
    )
    assert near.powered == 1.0
    far = graded(
        Placement("medium-electric-pole", 0, 0),
        Placement("assembling-machine-1", 9, 0, recipe="iron-gear-wheel"),
    )
    assert far.powered == 0.0


def test_burner_machines_are_not_counted_as_unpowered():
    assert graded(Placement("stone-furnace", 0, 0)).powered == 1.0
    assert graded(Placement("burner-inserter", 0, 0, SOUTH)).powered == 1.0


def test_a_belt_pointing_at_nothing_inside_the_design_is_a_mistake():
    # Two belts, the second facing back into the first: both lead somewhere.
    assert (
        graded(
            Placement("transport-belt", 0, 0, EAST),
            Placement("transport-belt", 1, 0, WEST),
        ).belts_lead_somewhere
        == 1.0
    )

    # A belt facing north with a lone assembler below it points into open space
    # that is inside the bounding box.
    report = graded(
        Placement("transport-belt", 0, 3, NORTH),
        Placement("transport-belt", 2, 0, EAST),
    )
    assert report.belts_lead_somewhere == 0.5


def test_a_belt_pointing_off_the_edge_is_an_output_not_an_error():
    assert graded(Placement("transport-belt", 0, 0, EAST)).belts_lead_somewhere == 1.0


def test_an_underground_entrance_is_judged_by_its_exit_not_its_next_tile():
    landed = graded(
        Placement("underground-belt", 0, 0, EAST, flow="input"),
        Placement("underground-belt", 4, 0, EAST, flow="output"),
    )
    assert landed.belts_lead_somewhere == 1.0

    # Past the five-tile reach of a yellow underground belt, the pair does not
    # connect and the entrance really is going nowhere.
    orphan = graded(
        Placement("underground-belt", 0, 0, EAST, flow="input"),
        Placement("underground-belt", 9, 0, EAST, flow="output"),
    )
    assert orphan.belts_lead_somewhere == 0.5


def test_quality_is_zero_for_anything_invalid():
    report = graded(
        Placement("assembling-machine-1", 0, 0),
        Placement("assembling-machine-1", 1, 1),
    )
    assert report.quality() == 0.0


def test_the_spec_is_honoured_within_tolerance_and_not_beyond_it():
    layout = [
        Placement("assembling-machine-2", index * 3, 0, recipe="iron-gear-wheel")
        for index in range(4)
    ]

    def honoured(count, width):
        spec = Spec("assembler-row", "iron-gear-wheel", count, width, 3)
        return graded(*layout, spec=spec).spec_honoured

    assert honoured(4, 12)
    assert honoured(6, 12)  # two machines out is within tolerance
    assert not honoured(9, 12)
    assert not honoured(4, 30)


def test_a_saturated_count_means_at_least_not_exactly():
    belts = [
        Placement("transport-belt", index, row, EAST) for row in range(2) for index in range(40)
    ]
    spec = Spec("belt-lane", None, grammar.COUNT_MAX, 40, 2)
    # 80 entities against a `#64` spec: the token cannot say more than 64.
    assert graded(*belts, spec=spec).spec_honoured


def test_grading_raw_text_reports_the_parse_failure_instead_of_raising():
    report = validate.grade("<bp> <e> not-a-thing x00 y00 d0 </bp>", DATA)
    assert not report.parsed
    assert not report.valid
    assert "unknown entity" in report.error


def test_summaries_aggregate_and_bucket_errors_by_shape():
    reports = [
        validate.grade("<bp> <e> not-a-thing x00 y00 d0 </bp>", DATA),
        validate.grade("<bp> <e> other-thing x00 y00 d0 </bp>", DATA),
        validate.grade("<bp> <e> transport-belt x00 y00 d2 </bp>", DATA),
    ]
    summary = validate.summarise(reports)
    assert summary.samples == 3
    assert summary.parse_rate == 1 / 3
    assert summary.valid_rate == 1 / 3
    # Two different unknown names collapse into one bar.
    assert list(summary.errors.values()) == [2]


def test_an_empty_batch_summarises_to_zero_rather_than_dividing_by_it():
    assert validate.summarise([]).samples == 0


def test_flow_is_reported_apart_from_quality_so_neither_can_hide_the_other():
    """The lesson of the measured run, as an assertion.

    Every local metric is at ceiling for this design — powered, both inserters
    connected, the belt leads somewhere — and it makes nothing, because the
    assembler is set to a recipe the belt does not carry.
    """
    starved = graded(
        # Facing west off the left edge, so every belt leads somewhere.
        Placement("transport-belt", 0, 0, WEST),
        Placement("inserter", 0, 1, SOUTH),
        Placement("assembling-machine-1", 0, 2, NORTH, recipe="electronic-circuit"),
        Placement("inserter", 0, 5, SOUTH),
        Placement("transport-belt", 0, 6, WEST),
        Placement("medium-electric-pole", 4, 3),
        spec=Spec(
            "module",
            "electronic-circuit",
            1,
            8,
            8,
            ports=(
                Port("in", "copper-plate", 0, 0, "n"),
                Port("out", "electronic-circuit", 0, 6, "s"),
            ),
        ),
    )
    assert starved.quality() == 1.0
    assert starved.flows() == 0.0
    assert starved.missing == ("electronic-circuit",)


def test_flow_is_zero_for_a_design_the_game_would_refuse():
    """For the same reason quality is: an overlap cannot be pasted, so what
    would have flowed through it is not a fact about anything."""
    report = graded(
        Placement("assembling-machine-1", 0, 0, recipe="copper-cable"),
        Placement("assembling-machine-1", 1, 1, recipe="copper-cable"),
    )
    assert not report.valid
    assert report.flows() == 0.0


def test_a_design_with_no_output_port_is_scored_on_being_fed_alone():
    """A belt lane has nothing to deliver, and averaging in a zero for failing to
    deliver it would make the flow number track the task mix instead of the model."""
    lane = graded(*[Placement("transport-belt", index, 0, EAST) for index in range(4)])
    assert not lane.ported
    assert lane.flows() == lane.fed


def test_summaries_average_flow_over_the_designs_that_have_any():
    good = validate.grade(
        "<bp> <e> transport-belt x00 y00 d2 <e> transport-belt x01 y00 d2 </bp>", DATA
    )
    broken = validate.grade("<bp> <e> not-a-thing x00 y00 d0 </bp>", DATA)
    summary = validate.summarise([good, broken])
    # Averaged over the one design that parsed: counting the unparseable one as
    # a flow failure would make flow a slower, noisier copy of validity.
    assert summary.mean_fed == good.fed
    assert summary.ported == 0
