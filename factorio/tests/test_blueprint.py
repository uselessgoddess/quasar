"""Blueprint strings, coordinates and the symmetry augmentations."""

import pytest

from quasar_factorio import blueprint as bp
from quasar_factorio import prototypes
from quasar_factorio.blueprint import Blueprint, BlueprintError, Placement
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST

DATA = prototypes.load()


def sample() -> Blueprint:
    return Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("assembling-machine-2", 1, 1, NORTH, recipe="iron-gear-wheel"),
            Placement("underground-belt", 0, 5, EAST, flow="input"),
            Placement("underground-belt", 4, 5, EAST, flow="output"),
        ]
    ).normalised(DATA)


def test_centre_positions_convert_to_top_left_tiles():
    # A 3x3 assembler centred on (10.5, 10.5) occupies tiles 9..11.
    payload = {
        "entities": [
            {"name": "assembling-machine-1", "position": {"x": 10.5, "y": 10.5}},
            {"name": "transport-belt", "position": {"x": 0.5, "y": 0.5}, "direction": 4},
        ]
    }
    got = bp.from_json(payload, DATA)
    by_name = {placement.name: placement for placement in got.entities}
    assert (by_name["transport-belt"].x, by_name["transport-belt"].y) == (0, 0)
    assert (by_name["assembling-machine-1"].x, by_name["assembling-machine-1"].y) == (9, 9)
    assert by_name["transport-belt"].direction == SOUTH


def test_a_half_tile_offset_design_keeps_all_of_its_rows():
    """Factorio 0.15 exported whole designs shifted half a tile.

    A 1x1 belt then sits at an integer position instead of x.5, so its corner
    lands exactly on .5 and the rounding rule decides whether the design
    survives. Python's `round` breaks ties towards even, which folds -1.5 and
    -2.5 onto the same tile — every second row of such a blueprint lands on the
    one before it and the whole thing reads as a pile of overlaps.
    """
    payload = {
        "entities": [
            {"name": "transport-belt", "position": {"x": float(x), "y": float(y)}}
            for y in (-2, -1, 0, 1)
            for x in (-2, -1, 0, 1)
        ]
    }

    got = bp.from_json(payload, DATA)

    assert len(got.entities) == 16
    assert len({(e.x, e.y) for e in got.entities}) == 16
    assert got.extent(DATA) == (4, 4)


def test_blueprint_string_round_trips():
    original = sample()
    text = bp.to_string(original, DATA, label="test")
    assert text.startswith("0")
    decoded = bp.from_json(bp.decode_string(text)["blueprint"], DATA, strict=True)
    assert decoded.entities == original.entities


def test_underground_flow_survives_the_round_trip():
    text = bp.to_string(sample(), DATA)
    payload = bp.decode_string(text)["blueprint"]
    tunnels = [e for e in payload["entities"] if e["name"] == "underground-belt"]
    # Both halves share one direction; only `type` tells them apart.
    assert {e["type"] for e in tunnels} == {"input", "output"}
    assert {e["direction"] for e in tunnels} == {EAST}


def test_unknown_prototypes_are_skipped_unless_strict():
    payload = {
        "entities": [
            {"name": "some-mod-machine", "position": {"x": 0.5, "y": 0.5}},
            {"name": "transport-belt", "position": {"x": 1.5, "y": 0.5}},
        ]
    }
    assert len(bp.from_json(payload, DATA).entities) == 1
    with pytest.raises(BlueprintError):
        bp.from_json(payload, DATA, strict=True)


def test_a_garbled_string_is_rejected_not_guessed_at():
    for text in ("", "not a blueprint", "0not-base64", "1eJw="):
        with pytest.raises(BlueprintError):
            bp.decode_string(text)


def test_four_rotations_return_the_original():
    original = sample()
    turned = original
    for _ in range(4):
        turned = bp.rotate(turned, 1, DATA)
    assert turned.entities == original.entities


def test_rotation_swaps_the_extent_and_turns_every_entity():
    original = sample()
    width, height = original.extent(DATA)
    turned = bp.rotate(original, 1, DATA)
    assert turned.extent(DATA) == (height, width)
    # Every direction advances one quarter turn: the belts were east, the
    # assembler was north.
    assert sorted(e.direction for e in turned.entities) == [EAST, SOUTH, SOUTH, SOUTH]


def test_rotation_never_introduces_an_overlap():
    from quasar_factorio.validate import overlapping

    original = sample()
    for quarters in range(4):
        assert overlapping(bp.rotate(original, quarters, DATA), DATA) == 0


def test_mirroring_is_its_own_inverse_and_swaps_east_for_west():
    original = sample()
    flipped = bp.mirror(original, DATA)
    # East becomes west; north is unmoved by a horizontal flip.
    assert sorted(e.direction for e in flipped.entities) == [NORTH, WEST, WEST, WEST]
    assert bp.mirror(flipped, DATA).entities == original.entities


def test_normalisation_is_idempotent_and_lands_on_the_origin():
    moved = Blueprint(entities=[Placement("transport-belt", 40, 30, EAST)])
    once = moved.normalised(DATA)
    assert (once.entities[0].x, once.entities[0].y) == (0, 0)
    assert once.normalised(DATA).entities == once.entities


def test_books_are_walked_recursively():
    payload = {
        "blueprint_book": {
            "blueprints": [
                {"blueprint": {"entities": []}},
                {"blueprint_book": {"blueprints": [{"blueprint": {"entities": [1]}}]}},
            ]
        }
    }
    assert len(list(bp.books(payload))) == 2
