"""The token grammar: what the model sees, and how strictly it is read back."""

import pytest

from quasar_factorio import grammar, prototypes
from quasar_factorio.blueprint import Blueprint, Placement
from quasar_factorio.grammar import ParseError, Spec
from quasar_factorio.prototypes import EAST, NORTH

DATA = prototypes.load()


def sample() -> Blueprint:
    return Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("assembling-machine-2", 0, 1, NORTH, recipe="iron-gear-wheel"),
            Placement("underground-belt", 0, 4, EAST, flow="output"),
        ]
    ).normalised(DATA)


def test_a_document_reads_the_way_it_is_documented():
    text = grammar.serialise(sample(), DATA, Spec("assembler-row", "iron-gear-wheel", 1, 3, 5))
    assert text.startswith(
        "<bp> <spec> k:assembler-row r:iron-gear-wheel #1 #3 #5 </spec>"
        " <e> transport-belt x00 y00 d2"
    )
    assert text.endswith("</bp>")
    assert " r:iron-gear-wheel" in text
    assert " t:output" in text


def test_serialise_parse_is_a_fixed_point():
    original = sample()
    spec = Spec.measure(original, "assembler-row", "iron-gear-wheel", DATA)
    text = grammar.serialise(original, DATA, spec)
    back, back_spec = grammar.parse(text, DATA)
    assert back.entities == original.entities
    assert back_spec == spec
    assert grammar.serialise(back, DATA, back_spec) == text


def test_the_spec_is_optional_on_both_sides():
    text = grammar.serialise(sample(), DATA)
    blueprint, spec = grammar.parse(text, DATA)
    assert spec is None
    assert len(blueprint.entities) == 3


def test_only_prototypes_that_need_them_carry_recipe_and_flow_tokens():
    text = grammar.serialise(sample(), DATA)
    tokens = text.split()
    # One recipe slot (the assembler) and one flow slot (the underground belt).
    assert sum(token.startswith("r:") for token in tokens) == 1
    assert sum(token in grammar.FLOWS for token in tokens) == 1
    # A belt gets neither, so its record is exactly five tokens.
    assert tokens[tokens.index("<e>") : tokens.index("<e>") + 5] == [
        "<e>",
        "transport-belt",
        "x00",
        "y00",
        "d2",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<e> transport-belt x00 y00 d0 </bp>",  # no <bp>
        "<bp> <e> transport-belt x00 y00 d0",  # no </bp>
        "<bp> <e> transport-belt x00 y00 </bp>",  # truncated entity
        "<bp> <e> nonsense-machine x00 y00 d0 </bp>",  # unknown prototype
        "<bp> <e> assembling-machine-1 x00 y00 d0 </bp>",  # missing recipe slot
        "<bp> <e> assembling-machine-1 x00 y00 d0 r:not-a-recipe </bp>",
        "<bp> <e> underground-belt x00 y00 d2 </bp>",  # missing flow slot
        "<bp> <e> transport-belt 0 0 d0 </bp>",  # bare digits, not coordinates
        "<bp> <e> transport-belt y00 x00 d0 </bp>",  # axes swapped
        "<bp> <spec> k:misc r:none #1 #2 </spec> </bp>",  # short spec
    ],
)
def test_malformed_streams_are_refused(text):
    with pytest.raises(ParseError):
        grammar.parse(text, DATA)


def test_a_parse_error_says_how_far_it_got():
    text = "<bp> <e> transport-belt x00 y00 d0 <e> transport-belt x01 y00 d0 <e> broken"
    with pytest.raises(ParseError) as caught:
        grammar.parse(text, DATA)
    assert caught.value.parsed == 2
    assert caught.value.at > 0


def test_a_leading_terminator_is_tolerated():
    text = grammar.serialise(sample(), DATA)
    blueprint, _ = grammar.parse(f"{grammar.EOS} {text}", DATA)
    assert len(blueprint.entities) == 3


def test_the_vocabulary_is_fixed_deduplicated_and_derived_from_the_table():
    vocab = grammar.vocabulary(DATA)
    assert vocab == grammar.vocabulary(DATA)
    assert len(vocab) == len(set(vocab))
    assert vocab[: len(grammar.STRUCTURAL)] == list(grammar.STRUCTURAL)
    assert vocab[0] == grammar.UNK and vocab[1] == grammar.EOS
    for name in DATA.entities:
        assert name in vocab
    for name in DATA.recipes:
        assert f"r:{name}" in vocab
    # It has to stay inside quasar's u16 shard vocabulary cap with room to spare.
    assert len(vocab) < 1024


def test_every_token_a_document_can_contain_is_in_the_vocabulary():
    vocab = set(grammar.vocabulary(DATA))
    text = grammar.serialise(sample(), DATA, Spec("mall-cell", "iron-gear-wheel", 1, 3, 5))
    assert set(text.split()) <= vocab


def test_spec_counts_crafting_machines_when_there_are_any():
    spec = Spec.measure(sample(), "assembler-row", "iron-gear-wheel", DATA)
    assert spec.count == 1  # one assembler, not three entities
    belts = Blueprint(entities=[Placement("transport-belt", index, 0, EAST) for index in range(4)])
    assert Spec.measure(belts.normalised(DATA), "belt-lane", None, DATA).count == 4


def test_counts_saturate_rather_than_wrapping():
    assert grammar.count_token(10_000) == f"#{grammar.COUNT_MAX}"
    assert grammar.count_token(-5) == "#0"
