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


def test_every_module_draw_makes_the_item_it_advertises():
    """The contract the other generators cannot be held to.

    A belt lane or a mall row is legal or it is not; a module additionally has to
    *work*, and that is checkable because the spec says what goes in and what
    should come out. Anything less than 1.0 here means the corpus is teaching a
    layout that starves, which is precisely the failure the local metrics missed.
    """
    for blueprint, spec in draw(synth.module, 120):
        report = validate.grade(grammar.serialise(blueprint, DATA, spec), DATA)
        assert report.ported, spec.product
        assert (report.delivers, report.fed, report.working) == (1.0, 1.0, 1.0), spec.product
        assert (report.mixed, report.leaks) == (0, 0), spec.product
        assert report.within_zone, spec.product


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
