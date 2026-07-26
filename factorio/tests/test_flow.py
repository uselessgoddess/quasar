"""The item-flow grader: does the design work, as opposed to parse?

Everything `validate` measures is local, which is why the measured run had every
one of its numbers at ceiling while nothing in the corpus was checked for making
the item it advertised. The blueprints here are hand-built and tiny — one belt,
one inserter, one machine, one inserter, one belt — because the interesting cases
are all one-entity edits of a working design: turn an inserter round, take the
output inserter away, feed the machine the wrong plate.
"""

from quasar_factorio import flow, prototypes
from quasar_factorio.blueprint import Blueprint, Placement
from quasar_factorio.grammar import Port, Spec
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST

DATA = prototypes.load()

# The smallest thing that is really a module: copper plate arrives on the top
# belt, an inserter puts it into an assembler making cable, another inserter
# takes the cable out onto the bottom belt. The assembler is 3x3 at (0,2), so it
# covers rows 2 to 4 and the second inserter reaches it from row 5.
IN_PORT = Port("in", "copper-plate", 0, 0, "n")
OUT_PORT = Port("out", "copper-cable", 0, 6, "s")


def module(*, out_inserter: bool = True, recipe: str = "copper-cable") -> Blueprint:
    entities = [
        Placement("transport-belt", 0, 0, EAST),
        Placement("inserter", 0, 1, SOUTH),
        Placement("assembling-machine-1", 0, 2, NORTH, recipe=recipe),
        Placement("transport-belt", 0, 6, EAST),
    ]
    if out_inserter:
        entities.insert(3, Placement("inserter", 0, 5, SOUTH))
    return Blueprint(entities=entities)


def spec(*ports: Port) -> Spec:
    return Spec(kind="module", product="copper-cable", width=8, height=8, ports=ports)


def traced(blueprint: Blueprint, *ports: Port) -> flow.Flow:
    return flow.trace(blueprint, spec(*ports) if ports else None, DATA)


def test_the_smallest_working_module_delivers_what_it_promises():
    report = traced(module(), IN_PORT, OUT_PORT)
    assert report.delivers == 1.0
    assert report.fed == 1.0
    assert report.working == 1.0
    assert report.missing == ()
    assert report.leaks == 0


def test_an_input_port_on_the_wrong_item_starves_the_machine():
    """The failure the local metrics cannot see.

    Every inserter is still connected, the machine is still powered, the belts
    still lead somewhere. It just makes nothing, forever.
    """
    report = traced(module(), Port("in", "iron-plate", 0, 0, "n"), OUT_PORT)
    assert report.fed == 0.0
    assert report.delivers == 0.0
    assert report.missing == ("copper-cable",)


def test_a_machine_with_no_output_inserter_is_fed_but_not_working():
    report = traced(module(out_inserter=False), IN_PORT, OUT_PORT)
    assert report.fed == 1.0
    assert report.working == 0.0
    assert report.delivers == 0.0


def test_an_inserter_facing_the_wrong_way_delivers_nothing():
    # Turned around, the input inserter picks up from the empty tile below it
    # and tries to put items onto the supply belt.
    turned = module()
    turned.entities[1] = Placement("inserter", 0, 1, NORTH)
    report = traced(turned, IN_PORT, OUT_PORT)
    assert report.fed == 0.0
    assert report.delivers == 0.0


def test_a_belt_running_into_an_assembler_is_not_a_delivery():
    """Belts feed belts. Counting the touch would let the model skip inserters."""
    beltfed = Blueprint(
        entities=[
            Placement("transport-belt", 0, 1, SOUTH),
            Placement("assembling-machine-1", 0, 2, NORTH, recipe="copper-cable"),
        ]
    )
    assert traced(beltfed, Port("in", "copper-plate", 0, 1, "n")).fed == 0.0


def test_items_cross_an_underground_pair_and_the_gap_is_not_a_dead_end():
    tunnelled = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("underground-belt", 1, 0, EAST, flow="input"),
            Placement("underground-belt", 5, 0, EAST, flow="output"),
            Placement("transport-belt", 6, 0, EAST),
        ]
    )
    report = flow.trace(tunnelled, spec(Port("in", "copper-plate", 0, 0, "w")), DATA)
    assert report.carried[3] == ("copper-plate",)


def test_an_unpaired_underground_belt_swallows_its_items():
    orphan = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("underground-belt", 1, 0, EAST, flow="input"),
            Placement("transport-belt", 6, 0, EAST),
        ]
    )
    report = flow.trace(orphan, spec(Port("in", "copper-plate", 0, 0, "w")), DATA)
    assert 2 not in report.carried


def test_a_belt_carrying_off_the_edge_at_no_declared_port_is_a_leak():
    leaky = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("transport-belt", 1, 0, EAST),
        ]
    )
    assert flow.trace(leaky, spec(Port("in", "copper-plate", 0, 0, "w")), DATA).leaks == 1
    # The same belt, declared as the output, is the module doing its job.
    declared = spec(Port("in", "copper-plate", 0, 0, "w"), Port("out", "copper-plate", 1, 0, "e"))
    assert flow.trace(leaky, declared, DATA).leaks == 0


def test_a_belt_over_two_item_types_is_mixed():
    junction = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("transport-belt", 1, 0, EAST),
        ]
    )
    ports = [Port("in", item, 0, 0, "w") for item in ("copper-plate", "iron-plate", "coal")]
    assert flow.trace(junction, spec(*ports), DATA).mixed == 2
    assert flow.trace(junction, spec(*ports[:2]), DATA).mixed == 0


def test_a_design_larger_than_the_zone_it_was_given_is_outside_it():
    report = traced(module(), IN_PORT, OUT_PORT)
    assert report.within_zone
    cramped = Spec(kind="module", width=3, height=3, ports=(IN_PORT, OUT_PORT))
    assert not flow.trace(module(), cramped, DATA).within_zone


def test_a_design_with_no_ports_has_its_inputs_guessed_from_the_edges():
    """So the metric stays comparable for the generators that predate ports.

    The belt at the top has nothing behind it, so it is an edge input, and what
    it carries is guessed from what the assembler wants and nothing here makes.
    """
    report = flow.trace(module(), None, DATA)
    assert report.fed == 1.0
    # Guessed inputs are never scored as a promise kept, because nothing promised.
    assert report.delivers == 0.0


def test_a_machine_only_accepts_what_its_recipe_calls_for():
    """An inserter would physically insert anything; that is a jam, not a feed."""
    wrong = module(recipe="iron-gear-wheel")
    report = traced(wrong, IN_PORT, Port("out", "iron-gear-wheel", 0, 6, "s"))
    assert report.fed == 0.0
    assert report.carried.get(2) is None


def test_the_score_pays_more_for_delivering_than_for_being_fed():
    fedonly = traced(module(out_inserter=False), IN_PORT, OUT_PORT)
    whole = traced(module(), IN_PORT, OUT_PORT)
    assert whole.score() == 1.0
    assert 0.0 < fedonly.score() < 0.5


def test_an_empty_design_scores_zero_rather_than_dividing_by_nothing():
    report = flow.trace(Blueprint(), None, DATA)
    assert (report.delivers, report.fed, report.working) == (0.0, 0.0, 0.0)


def test_a_two_stage_chain_hands_its_intermediate_down_the_stack():
    """The layout the whole planner exists for, at its smallest.

    Copper plate on the top belt becomes cable in the first assembler, lands on
    the middle belt, and is taken with iron plate into circuits below. The middle
    belt carries both, which is exactly the two lanes a belt has.
    """
    stacked = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("inserter", 0, 1, SOUTH),
            Placement("assembling-machine-1", 0, 2, NORTH, recipe="copper-cable"),
            Placement("inserter", 0, 5, SOUTH),
            Placement("transport-belt", 0, 6, EAST),
            Placement("inserter", 0, 7, SOUTH),
            Placement("assembling-machine-1", 0, 8, NORTH, recipe="electronic-circuit"),
            Placement("inserter", 0, 11, SOUTH),
            Placement("transport-belt", 0, 12, EAST),
        ]
    )
    ports = (
        Port("in", "copper-plate", 0, 0, "n"),
        Port("in", "iron-plate", 0, 6, "w"),
        Port("out", "electronic-circuit", 0, 12, "s"),
    )
    report = flow.trace(stacked, Spec(kind="module", width=8, height=16, ports=ports), DATA)
    assert report.delivers == 1.0
    assert report.fed == 1.0
    assert report.working == 1.0
    assert report.mixed == 0


def test_the_pickup_and_insert_tables_are_opposites():
    for direction in (NORTH, EAST, SOUTH, WEST):
        ahead, back = flow.AHEAD[direction], flow.PICKUP[direction]
        assert (ahead[0] + back[0], ahead[1] + back[1]) == (0, 0)
