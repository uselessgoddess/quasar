"""The recipe arithmetic. If this is wrong, every module starves politely.

The planner's mistakes are the expensive kind: they produce a blueprint that
parses, fits, is powered, and has every inserter facing something — and then
never makes the item it advertises. So these tests check the arithmetic against
recipes stated in `assets/prototypes.json`, and they check the four filters that
decide which chains are allowed to become training documents at all.
"""

import pytest

from quasar_factorio import plan, prototypes

DATA = prototypes.load()

PLATES = ("iron-plate", "copper-plate")


def test_the_flagship_chain_is_two_stages_in_build_order():
    # The example from the issue: mark a zone, feed it plates, ask for circuits.
    unit = plan.solve(DATA, "electronic-circuit", PLATES, rate=1.0)
    assert [stage.product for stage in unit.stages] == ["copper-cable", "electronic-circuit"]
    # Build order, so the layout can put each stage under the one that feeds it.
    assert unit.stages[0].ingredients == ("copper-plate",)
    assert set(unit.stages[1].ingredients) == {"iron-plate", "copper-cable"}


def test_stages_come_back_topologically_sorted():
    """Whatever the chain, a stage is never listed before something it consumes.

    This is weaker than the linearity the layout needs — a stage may want the
    product of the stage two rows up — and that gap is exactly what the module
    catalogue filters out below.
    """
    for product, supply in plan.chains(DATA, depth=3).items():
        unit = plan.solve(DATA, product, supply, rate=1.0, depth=3)
        made = set(unit.supply)
        for stage in unit.stages:
            assert set(stage.ingredients) <= made, (product, stage.recipe)
            made.add(stage.product)


def test_demand_flows_backwards_and_rounds_up_to_whole_machines():
    # An assembling-machine-1 runs at speed 0.5 and both recipes take 0.5s, so
    # it turns out one circuit a second, or two copper cable. One circuit a
    # second eats three cable a second, which is one and a half machines — and
    # half an assembler cannot be placed, so it rounds up to two.
    unit = plan.solve(DATA, "electronic-circuit", PLATES, rate=1.0)
    cable, circuit = unit.stages
    assert (circuit.count, cable.count) == (1, 2)
    # And the reported rate is what the rounded-up count really makes, not the
    # fractional ideal a throughput metric would then measure against.
    assert cable.rate == pytest.approx(4.0)
    assert unit.rate == pytest.approx(1.0)


def test_an_item_that_is_neither_supplied_nor_craftable_is_refused():
    with pytest.raises(plan.PlanError):
        plan.solve(DATA, "electronic-circuit", ("iron-plate",))
    with pytest.raises(plan.PlanError):
        plan.solve(DATA, "iron-plate", PLATES)  # already supplied, nothing to do
    with pytest.raises(plan.PlanError):
        plan.solve(DATA, "no-such-item", PLATES)


def test_depth_is_a_wall_not_a_preference():
    # `depth` counts hops below the product, so a rail signal — cable, then
    # circuit, then the signal — needs two and is refused at one rather than
    # quietly coming back as something shallower.
    assert [stage.product for stage in plan.solve(DATA, "rail-signal", PLATES, depth=2).stages] == [
        "copper-cable",
        "electronic-circuit",
        "rail-signal",
    ]
    with pytest.raises(plan.PlanError):
        plan.solve(DATA, "rail-signal", PLATES, depth=1)


def test_fit_spends_the_whole_machine_budget_and_never_overspends():
    small = plan.fit(DATA, "electronic-circuit", PLATES, machines=4)
    large = plan.fit(DATA, "electronic-circuit", PLATES, machines=12)
    assert small.machines <= 4
    assert large.machines <= 12
    assert large.machines > small.machines
    assert large.rate > small.rate

    with pytest.raises(plan.PlanError):
        # One machine a stage is the floor; a two-stage chain cannot be smaller.
        plan.fit(DATA, "electronic-circuit", PLATES, machines=1)


def test_the_module_catalogue_is_not_empty_and_leads_with_the_flagship():
    catalogue = plan.modules(DATA)
    assert len(catalogue) >= 10
    assert plan.Module("electronic-circuit", ("iron-plate", "copper-plate"), 2) in catalogue


def test_every_catalogued_module_survives_its_own_four_filters():
    for module in plan.modules(DATA):
        unit = plan.solve(DATA, module.product, module.supply, rate=1e-9, depth=module.depth)
        # Two stages or more: one machine on a belt is not a module.
        assert len(unit.stages) >= 2, module
        # Every machine can be told what to make. A furnace cannot, so a chain
        # through one would be a blueprint the grammar has no way to express.
        assert all(DATA.entities[stage.machine].takes_recipe for stage in unit.stages), module
        # At most two items on any belt: a transport belt has two lanes.
        for rank, stage in enumerate(unit.stages):
            assert len(unit.raws_of(stage)) + bool(rank) <= 2, module
        # Linear: each stage is fed by the supply and by the stage just above.
        reachable = set(unit.supply)
        for stage in unit.stages:
            assert set(stage.ingredients) <= reachable, module
            reachable = set(unit.supply) | {stage.product}


def test_a_catalogued_supply_never_names_an_item_no_stage_consumes():
    """A port for an item nothing takes is a leak, and a lesson in ignoring ports.

    `arithmetic-combinator` reaches copper cable by two paths of different
    lengths, so the leaf set names copper plate even though the solved plan stops
    at cable and never asks for the plate. Declaring that port would train the
    model that an input in the prompt need not be connected to anything.
    """
    for module in plan.modules(DATA):
        unit = plan.solve(DATA, module.product, module.supply, rate=1e-9, depth=module.depth)
        wanted = {item for stage in unit.stages for item in stage.ingredients}
        assert set(module.supply) <= wanted, module


def test_depth_travels_with_the_module_because_it_changes_what_gets_built():
    """The same product from the same supply is a different module at each depth.

    `modules` offers every boundary from two stages up, so the catalogue can hold
    both a shallow design that is handed its intermediate and a deep one that
    makes it. Solving one at the other's depth silently returns the other.
    """
    catalogue = plan.modules(DATA)
    assert {module.depth for module in catalogue} > {2}
    for module in catalogue:
        unit = plan.solve(DATA, module.product, module.supply, rate=1e-9, depth=module.depth)
        assert len(unit.stages) <= module.depth
        # The pair is unique: two entries never disagree about how deep they are.
        twins = [
            other
            for other in catalogue
            if (other.product, other.supply) == (module.product, module.supply)
        ]
        assert twins == [module]


def test_a_wider_depth_offers_strictly_more_modules():
    assert set(plan.modules(DATA, depth=2)) < set(plan.modules(DATA, depth=3))


def test_a_one_lane_catalogue_is_a_subset_of_a_two_lane_one():
    assert set(plan.modules(DATA, lanes=1)) <= set(plan.modules(DATA, lanes=2))
