"""Augmentations have to be symmetries, not noise."""

import random

import pytest

from quasar_factorio import augment, grammar, prototypes, synth, validate
from quasar_factorio.blueprint import Blueprint, Placement
from quasar_factorio.prototypes import EAST, NORTH

DATA = prototypes.load()


def test_every_upgrade_keeps_the_footprint_and_never_shortens_reach():
    """The table is a claim about the game; this is the claim checked.

    A tier swap that changed a 2x2 into a 3x3 would silently overlap its
    neighbour, and one that shortened an underground belt's reach would sever
    tunnels the original had connected.
    """
    for before, after in augment.UPGRADES.items():
        old, new = DATA.entities[before], DATA.entities[after]
        assert (old.width, old.height) == (new.width, new.height), before
        assert (old.max_distance or 0) <= (new.max_distance or 0), before
        assert set(old.crafting_categories) <= set(new.crafting_categories), before
        assert old.takes_recipe == new.takes_recipe, before
        assert old.takes_flow == new.takes_flow, before


def test_retier_moves_both_halves_of_a_tunnel_together():
    pair = Blueprint(
        entities=[
            Placement("underground-belt", 0, 0, EAST, flow="input"),
            Placement("underground-belt", 4, 0, EAST, flow="output"),
        ]
    ).normalised(DATA)
    upgraded = augment.retier(pair, DATA)
    assert {entity.name for entity in upgraded.entities} == {"fast-underground-belt"}
    assert {entity.flow for entity in upgraded.entities} == {"input", "output"}


def test_retier_declines_when_there_is_nothing_to_upgrade():
    poles = Blueprint(entities=[Placement("medium-electric-pole", 0, 0)]).normalised(DATA)
    assert augment.retier(poles, DATA) is None


def test_the_dihedral_group_has_eight_members_in_a_fixed_order():
    blueprint, _ = synth.GENERATORS["mall-cell"](random.Random(4), DATA)
    forms = list(augment.dihedral(blueprint, DATA))
    assert len(forms) == 8
    assert forms[0].entities == blueprint.normalised(DATA).entities
    again = list(augment.dihedral(blueprint, DATA))
    assert [form.entities for form in forms] == [form.entities for form in again]


def test_the_original_orientation_is_always_the_first_variant():
    blueprint, spec = synth.GENERATORS["assembler-row"](random.Random(2), DATA)
    forms = augment.variants(blueprint, spec, DATA, rng=random.Random(9), limit=3)
    assert forms[0][0].entities == blueprint.normalised(DATA).entities
    assert forms[0][1] == spec


def test_variants_are_distinct_and_bounded_by_the_limit():
    blueprint, spec = synth.GENERATORS["smelter-column"](random.Random(5), DATA)
    forms = augment.variants(blueprint, spec, DATA, rng=random.Random(1), limit=6)
    assert len(forms) == 6
    assert len({tuple(form.entities) for form, _ in forms}) == 6


def test_the_spec_is_remeasured_rather_than_carried_over():
    """A rotated 3x9 column is 9x3, and the prompt has to say so."""
    blueprint, spec = synth.GENERATORS["smelter-column"](random.Random(0), DATA)
    for form, form_spec in augment.variants(blueprint, spec, DATA, rng=random.Random(3), limit=8):
        assert (form_spec.width, form_spec.height) == form.extent(DATA)
        assert form_spec.kind == spec.kind and form_spec.product == spec.product


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_augmenting_never_degrades_a_design(kind):
    reports = []
    for seed in range(20):
        rng = random.Random(seed)
        blueprint, spec = synth.GENERATORS[kind](rng, DATA)
        for form, form_spec in augment.variants(blueprint, spec, DATA, rng=rng, limit=8):
            reports.append(validate.grade(grammar.serialise(form, DATA, form_spec), DATA))
    summary = validate.summarise(reports)
    assert summary.valid_rate == 1.0, summary.errors
    assert summary.spec_rate == 1.0
    assert summary.mean_powered == 1.0
    assert summary.mean_connected == 1.0
    assert summary.mean_belts == 1.0


def test_a_rotated_module_still_delivers_through_its_rotated_ports():
    """The reason a port is a tile and a side rather than a name for an edge.

    Rotate the design and the ports have to follow it exactly, or the prompt now
    promises iron plate arriving somewhere the belt no longer is — and the
    document teaches the model that the ports in its prompt mean nothing.
    """
    for seed in range(12):
        rng = random.Random(seed)
        blueprint, spec = synth.module(rng, DATA)
        for form, form_spec in augment.variants(blueprint, spec, DATA, rng=rng, limit=8):
            report = validate.grade(grammar.serialise(form, DATA, form_spec), DATA)
            assert (report.delivers, report.fed) == (1.0, 1.0), (seed, form_spec.product)
            assert report.within_zone


def test_a_symmetric_design_collapses_instead_of_duplicating():
    """A straight belt lane has four orientations, not eight.

    Mirroring an east-facing row gives a west-facing row, which the rotations
    already produced — so the group orbit is half the size, and the deduplication
    has to notice rather than writing the same document twice.
    """
    lane = Blueprint(
        entities=[Placement("transport-belt", index, 0, EAST) for index in range(4)]
    ).normalised(DATA)
    spec = grammar.Spec.measure(lane, "belt-lane", None, DATA)
    assert len({tuple(form.entities) for form in augment.dihedral(lane, DATA)}) == 4
    # The eight the augmenter does offer are the four turns of each belt tier.
    forms = augment.variants(lane, spec, DATA, rng=random.Random(0), limit=16)
    assert len(forms) == 8
    assert len({tuple(form.entities) for form, _ in forms}) == 8


def test_variants_that_would_not_fit_the_grid_are_dropped():
    tall = Blueprint(
        entities=[Placement("transport-belt", index, 0, NORTH) for index in range(64)]
    ).normalised(DATA)
    spec = grammar.Spec.measure(tall, "belt-lane", None, DATA)
    for form, _ in augment.variants(tall, spec, DATA, rng=random.Random(0), limit=8):
        assert form.fits(DATA)
