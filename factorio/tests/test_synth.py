"""The synthetic generators, held to the standard they teach.

A generator that emits subtly broken factories teaches the model to emit subtly
broken factories, so every template is drawn a thousand times and has to come
back valid *and* well connected. These thresholds are the contract; loosening one
to make a build pass is loosening the corpus.
"""

import random

import pytest

from quasar_factorio import augment, flow, grammar, plan, prototypes, synth, validate
from quasar_factorio.blueprint import GRID

DATA = prototypes.load()
DRAWS = 200


def draw(generator, count=DRAWS):
    for seed in range(count):
        yield generator(random.Random(seed), DATA)


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_every_draw_is_valid_and_connected(kind):
    reports = []
    for blueprint, spec in draw(synth.GENERATORS[kind]):
        # Graded through the same path a generation takes, tokens and all.
        reports.append(validate.grade(grammar.serialise(blueprint, DATA, spec), DATA))
    summary = validate.summarise(reports)
    assert summary.valid_rate == 1.0, summary.errors
    assert summary.spec_rate == 1.0
    assert summary.mean_powered == 1.0
    assert summary.mean_connected == 1.0
    assert summary.mean_belts == 1.0
    assert summary.mean_entities >= 4


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_every_draw_fits_the_grid(kind):
    for blueprint, _ in draw(synth.GENERATORS[kind], 60):
        width, height = blueprint.extent(DATA)
        assert 0 < width <= GRID and 0 < height <= GRID


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_no_draw_overflows_the_context(kind):
    """Every document, plus its terminator, has to fit `seq_len`.

    Measured on the real token stream rather than estimated from the entity
    count: a truncated document teaches the model that `</bp>` is optional,
    which is exactly the failure mode that ruins the validity metric.
    """
    for blueprint, spec in draw(synth.GENERATORS[kind], 300):
        assert len(grammar.serialise(blueprint, DATA, spec).split()) + 1 <= 512


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_generators_are_deterministic_given_a_seed(kind):
    generator = synth.GENERATORS[kind]
    first, spec = generator(random.Random(7), DATA)
    second, spec2 = generator(random.Random(7), DATA)
    assert first.entities == second.entities
    assert spec == spec2


def test_an_explicit_module_target_uses_the_same_deterministic_draw_path():
    target = plan.modules(DATA)[7]
    first = synth.module_for(random.Random(23), DATA, target)
    second = synth.module_for(random.Random(23), DATA, target)
    assert first == second
    assert first[1].product == target.product


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_generators_actually_vary(kind):
    shapes = {blueprint.extent(DATA) for blueprint, _ in draw(synth.GENERATORS[kind], 40)}
    assert len(shapes) > 1


def test_the_spec_kind_matches_the_generator_that_made_it():
    for kind, generator in synth.GENERATORS.items():
        _, spec = generator(random.Random(1), DATA)
        assert spec.kind == kind
        assert kind in grammar.KINDS


def test_every_draw_survives_a_trip_through_a_real_blueprint_string():
    from quasar_factorio import blueprint as bp

    for index in range(120):
        original, spec = synth.sample(random.Random(index), DATA)
        text = grammar.serialise(original, DATA, spec)
        payload = bp.decode_string(bp.to_string(original, DATA))["blueprint"]
        again = bp.from_json(payload, DATA, strict=True)
        assert grammar.serialise(again, DATA, spec) == text


def test_sampling_reaches_every_generator():
    rng = random.Random(0)
    kinds = {synth.sample(rng, DATA)[1].kind for _ in range(400)}
    assert kinds == set(synth.GENERATORS)


def test_recipes_are_only_ever_paired_with_machines_that_can_run_them():
    for index in range(300):
        blueprint, _ = synth.sample(random.Random(index), DATA)
        assert validate.illegal_recipes(blueprint, DATA) == 0


FORKS = tuple(module for module in plan.modules(DATA) if module.shape == "fork")
FACTORIES = tuple(module for module in plan.modules(DATA) if module.shape == "factory")


def test_every_module_draw_makes_the_item_it_advertises():
    """The contract the other generators cannot be held to.

    A belt lane or a mall row is legal or it is not; a module additionally has to
    *work*, and that is checkable because the spec says what goes in and what
    should come out. Anything less than 1.0 here means the corpus is teaching a
    layout that starves, which is precisely the failure the local metrics missed.

    Both shapes go through this loop, and the products are collected so that the
    branching ones are covered on purpose rather than by luck: a fork is the
    layout most likely to starve, because its last machine is fed by two columns
    and being wrong about either one of them is enough.
    """
    products = set()
    for blueprint, spec in draw(synth.module, 200):
        report = validate.grade(grammar.serialise(blueprint, DATA, spec), DATA)
        assert report.ported, spec.product
        assert (report.delivers, report.fed, report.working) == (1.0, 1.0, 1.0), spec.product
        assert (report.mixed, report.leaks) == (0, 0), spec.product
        assert report.within_zone, spec.product
        products.add(spec.product)
    assert {module.product for module in FORKS} <= products
    assert products - {module.product for module in FORKS}  # and stacks too


def test_factories_route_shared_intermediates_and_three_item_stages():
    assert {target.product for target in FACTORIES} == {
        "logistic-science-pack",
        "power-switch",
    }
    for target in FACTORIES:
        for seed in range(12):
            blueprint, spec = synth.module_for(random.Random(seed), DATA, target)
            report = validate.grade(grammar.serialise(blueprint, DATA, spec), DATA)
            assert (report.delivers, report.fed, report.working) == (1.0, 1.0, 1.0)
            assert (report.mixed, report.leaks) == (0, 0)
            if target.product == "logistic-science-pack":
                assert len([port for port in spec.inputs() if port.item == "iron-plate"]) >= 2


def test_factory_targets_have_many_geometrically_distinct_training_layouts():
    """Each DAG must be a family of routes, not templates seen under rotations."""
    for target in FACTORIES:
        layouts = set()
        for seed, form in enumerate(synth.FACTORY_FORMS):
            blueprint, spec = synth.module_for(random.Random(seed), DATA, target, factory_form=form)
            report = validate.grade(grammar.serialise(blueprint, DATA, spec), DATA)
            assert (report.delivers, report.fed, report.working) == (1.0, 1.0, 1.0)
            assert (report.mixed, report.leaks) == (0, 0)
            layouts.add(augment.canonical(blueprint, DATA))
        assert len(layouts) == len(synth.FACTORY_FORMS) == 32


def test_a_branching_module_is_two_columns_and_not_a_deeper_stack():
    """The geometry the flow report cannot tell apart from a lucky stack.

    A fork exists because its last machine wants two made items, and a single
    column of bands can deliver only one of them — but it also exists to keep the
    lanes honest, and that part is geometry: two branches with two sets of raws
    need two belts per row, not one wide one carrying the union. So every band
    row of a fork holds a pair of runs pointed at each other, one fed from each
    edge, ending nose to nose against the pole between them.

    Read off the builder's own output rather than a draw from `module`, which
    rotates what it is given and would turn the rows into columns.
    """
    for index, target in enumerate(FORKS):
        for seed in range(6):
            rng = random.Random(seed * 31 + index)
            blueprint, ports, _ = synth._forked(rng, DATA, target, synth.Layout.draw(rng))
            rows = {}
            for placement in blueprint.entities:
                if DATA.entity(placement.name).category in flow.BELTS:
                    rows.setdefault(placement.y, set()).add(placement.direction)
            assert any(len(runs) > 1 for runs in rows.values()), (target, seed)
            # One way in per branch per band, and exactly one way out.
            assert [port.role for port in ports].count("out") == 1, target
            assert {port.side for port in ports if port.role == "in"} == {"w", "e"}, target


def test_a_module_declares_a_port_for_everything_it_is_handed_and_makes():
    for _, spec in draw(synth.module, 60):
        assert spec.outputs()
        assert {port.item for port in spec.outputs()} == {spec.product}
        # Every input port names an item some stage of the plan actually takes,
        # so no port in the prompt is a promise the design cannot use.
        wanted = set()
        for step in spec.plan:
            recipe = DATA.recipes[step.recipe]
            wanted |= {name for name, _ in recipe.ingredients}
        assert {port.item for port in spec.inputs()} <= wanted


def test_a_module_port_sits_on_a_belt_at_the_edge_it_names():
    edges = {"n": lambda x, y, w, h: y == 0, "s": lambda x, y, w, h: y == h - 1}
    edges |= {"w": lambda x, y, w, h: x == 0, "e": lambda x, y, w, h: x == w - 1}
    for blueprint, spec in draw(synth.module, 60):
        width, height = blueprint.extent(DATA)
        occupied = {
            tile: placement for placement in blueprint.entities for tile in placement.tiles(DATA)
        }
        for port in spec.ports:
            assert edges[port.side](port.x, port.y, width, height), port
            placement = occupied.get((port.x, port.y))
            assert placement is not None, port
            assert DATA.entity(placement.name).category in flow.BELTS, port


def test_the_module_generator_does_not_run_out_of_layouts():
    """Counted the way the corpus builder counts, which is the only way that pays.

    A generator that draws the same design twice contributes it once: the build
    deduplicates on `augment.canonical`, which sees through rotation, reflection
    and tier. Before `synth.Layout` the module generator varied nothing else, so
    twenty catalogue entries times a couple of machine budgets was the whole of
    it — forty-five designs, and the thousandth draw added nothing to the corpus
    that the fiftieth had not. This is the assertion that keeps that from
    quietly coming back; `experiments/module_yield.py` prints the curve.
    """
    keys = {augment.canonical(blueprint, DATA) for blueprint, _ in draw(synth.module, 200)}
    assert len(keys) > 120


def test_a_module_plan_is_the_planner_s_plan_and_not_a_guess():
    for _, spec in draw(synth.module, 40):
        supply = {port.item for port in spec.inputs()}
        # Matched as a set: the ports come out in layout order, which is the
        # order the belts are stacked in and not the order the catalogue lists.
        catalogued = [
            module
            for module in plan.modules(DATA)
            if module.product == spec.product and set(module.supply) == supply
        ]
        assert catalogued, (spec.product, supply)
        machines = sum(step.count for step in spec.plan)
        unit = plan.fit(
            DATA,
            spec.product,
            catalogued[0].supply,
            machines=machines,
            depth=catalogued[0].depth,
        )
        assert unit.steps() == spec.plan
