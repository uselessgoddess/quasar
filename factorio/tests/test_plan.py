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


def test_a_science_pack_is_belted_and_a_fluid_is_not():
    """The one that was wrong, and the reason the catalogue had no science in it.

    `is_fluid` used to mean "has no stack size", and the distilled item table
    only held `data.raw.item` — so a science pack, which Factorio files under
    `tool`, was a fluid. Nothing raised: `chains` skips fluids silently, so the
    entire science branch was missing from the catalogue and read as a chain the
    layout filters had rejected.
    """
    for pack in ("automation-science-pack", "logistic-science-pack"):
        assert not plan.is_fluid(DATA, pack)
        assert plan.chains(DATA, depth=3).get(pack) is not None, pack
    for fluid in ("water", "petroleum-gas", "lubricant"):
        assert plan.is_fluid(DATA, fluid)
    # A name the table has never heard of still fails closed: better to refuse a
    # chain than to route a typo onto a belt.
    assert plan.is_fluid(DATA, "no-such-item")


def test_the_module_catalogue_is_not_empty_and_leads_with_the_flagship():
    catalogue = plan.modules(DATA)
    assert len(catalogue) >= 10
    assert plan.Module("electronic-circuit", ("iron-plate", "copper-plate"), 2) in catalogue


def banded(unit, stages):
    """The one condition both layouts are built on, restated for the tests.

    A column of bands: every stage fed by the supply and by the stage just above
    it, and no belt carrying more than the two lanes a transport belt has.
    """
    reachable = set(unit.supply)
    for rank, stage in enumerate(stages):
        assert set(stage.ingredients) <= reachable, stage.recipe
        assert len(unit.raws_of(stage)) + bool(rank) <= 2, stage.recipe
        reachable = set(unit.supply) | {stage.product}


def test_every_catalogued_module_survives_its_own_filters():
    for module in plan.modules(DATA):
        unit = plan.solve(DATA, module.product, module.supply, rate=1e-9, depth=module.depth)
        # Two stages or more: one machine on a belt is not a module.
        assert len(unit.stages) >= 2, module
        # Every machine can be told what to make. A furnace cannot, so a chain
        # through one would be a blueprint the grammar has no way to express.
        assert all(DATA.entities[stage.machine].takes_recipe for stage in unit.stages), module
        if module.shape == "stack":
            assert plan.fork(unit) is None, module
            banded(unit, unit.stages)
        else:
            # A fork is two columns of bands and a machine that takes both.
            left, right = plan.fork(unit)
            assert module.shape == "fork", module
            banded(unit, left)
            banded(unit, right)
            assert set(unit.stages[-1].ingredients) == {left[-1].product, right[-1].product}
            assert unit.raws_of(unit.stages[-1]) == ()


def test_the_catalogue_holds_at_least_five_branching_chains():
    """The point of forks: chains the stacked layout could not express at all.

    Every one of these has a last machine wanting two things that both have to
    be made, which is precisely what a run of bands cannot deliver — the first
    of the two would sail past the row that wants it.
    """
    forks = [module for module in plan.modules(DATA, depth=4) if module.shape == "fork"]
    assert len(forks) >= 5
    assert {module.product for module in forks} >= {"boiler", "repair-pack"}
    for module in forks:
        unit = plan.solve(DATA, module.product, module.supply, rate=1e-9, depth=module.depth)
        # Not linear, or the stacked layout would have claimed it first.
        reachable = set(unit.supply)
        linear = True
        for stage in unit.stages:
            linear = linear and set(stage.ingredients) <= reachable
            reachable = set(unit.supply) | {stage.product}
        assert not linear, module


def test_green_science_is_still_a_factory_and_for_two_reasons():
    """The honest limit, stated as a test so it cannot be lost by accident.

    Green science is the chain everyone reaches for, and a fork is exactly the
    right shape for its last machine: an inserter and a transport belt, both
    made, into a lab pack. It is still refused, twice over. The gear stage feeds
    both branches, so they are not two independent columns but a diamond; and
    the inserter wants circuits, gears *and* iron plate, which is three item
    types on a belt that has two lanes. Neither is fixed by branching — the
    first needs a shared intermediate routed sideways, the second needs a second
    belt per machine — so `logistic-science-pack` stays out of the catalogue
    rather than going in as a layout that starves.
    """
    unit = plan.solve(DATA, "logistic-science-pack", PLATES, rate=1e-9, depth=4)
    final, inserter = unit.stages[-1], unit.stages[-3]
    assert set(final.ingredients) == {"inserter", "transport-belt"}
    assert unit.raws_of(final) == ()  # the convergence belt itself is fine
    assert len(set(inserter.ingredients)) == 3  # the belt above the inserter is not
    assert plan.fork(unit) is None
    assert "logistic-science-pack" not in {module.product for module in plan.modules(DATA, depth=4)}


def test_one_lane_is_not_enough_for_any_fork():
    """Two branch products are two item types, so a one-lane belt has no fork."""
    for module in plan.modules(DATA, lanes=1):
        assert module.shape == "stack", module


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
        # Depth bounds the longest path to the supply, not the stage count: a
        # fork's two branches are the same distance down and are counted once.
        split = plan.fork(unit)
        longest = max(len(side) for side in split) + 1 if split else len(unit.stages)
        assert longest <= module.depth
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
