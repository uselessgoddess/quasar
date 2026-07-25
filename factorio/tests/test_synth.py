"""The synthetic generators, held to the standard they teach.

A generator that emits subtly broken factories teaches the model to emit subtly
broken factories, so every template is drawn a thousand times and has to come
back valid *and* well connected. These thresholds are the contract; loosening one
to make a build pass is loosening the corpus.
"""

import random

import pytest

from quasar_factorio import grammar, prototypes, synth, validate
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
