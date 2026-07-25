"""The renderer is the only way anyone will ever look at model output.

So the tests read pixels back rather than checking that a function returned
something: a picture that is silently blank, or a contact sheet whose cards
overlap, still passes every assertion about return types.
"""

import random
import struct
import xml.etree.ElementTree as ElementTree
import zlib

import pytest

from quasar_factorio import prototypes, render, synth
from quasar_factorio.blueprint import GRID, Blueprint, Placement
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST

DATA = prototypes.load()


def decode(blob: bytes) -> tuple[int, int, list[tuple[int, int, int]]]:
    """A PNG back to (width, height, pixels), with no third-party decoder.

    Deliberately not `render`'s own code in reverse: the point is to prove the
    bytes are a PNG a real decoder would accept, so this walks the chunk
    structure the way one would.
    """
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    at, chunks = 8, {}
    order = []
    while at < len(blob):
        (length,) = struct.unpack(">I", blob[at : at + 4])
        kind = blob[at + 4 : at + 8]
        body = blob[at + 8 : at + 8 + length]
        (checksum,) = struct.unpack(">I", blob[at + 8 + length : at + 12 + length])
        assert checksum == zlib.crc32(kind + body), kind
        chunks.setdefault(kind, b"")
        chunks[kind] += body
        order.append(kind)
        at += 12 + length

    assert order[0] == b"IHDR" and order[-1] == b"IEND"
    width, height, depth, colour, *rest = struct.unpack(">IIBBBBB", chunks[b"IHDR"])
    assert (depth, colour, rest) == (8, 2, [0, 0, 0])

    raw = zlib.decompress(chunks[b"IDAT"])
    stride = width * 3
    pixels = []
    for line in range(height):
        start = line * (stride + 1)
        assert raw[start] == 0, "only filter type 0 is emitted"
        row = raw[start + 1 : start + 1 + stride]
        pixels += [tuple(row[at : at + 3]) for at in range(0, stride, 3)]
    assert len(pixels) == width * height
    return width, height, pixels


def pixel(canvas: render.Raster, x: int, y: int) -> tuple[int, int, int]:
    at = (y * canvas.width + x) * 3
    return tuple(canvas.pixels[at : at + 3])


# --- the framebuffer -------------------------------------------------------


def test_a_png_round_trips_through_a_hand_written_decoder():
    canvas = render.Raster(6, 4, fill=render.GROUND)
    canvas.box(1, 1, 2, 2, render.ALERT)
    width, height, pixels = decode(canvas.png())
    assert (width, height) == (6, 4)
    assert pixels[0] == render.GROUND
    assert pixels[1 * 6 + 1] == render.ALERT
    assert pixels[2 * 6 + 2] == render.ALERT
    assert pixels[1 * 6 + 3] == render.GROUND


@pytest.mark.parametrize(
    "x,y,w,h",
    [(-4, -4, 3, 3), (100, 0, 5, 5), (-2, 1, 4, 1), (5, -1, 1, 4), (0, 0, 0, 0)],
)
def test_drawing_outside_the_canvas_is_clipped_not_wrapped(x, y, w, h):
    canvas = render.Raster(8, 8)
    before = bytes(canvas.pixels)
    canvas.box(x, y, w, h, render.PAPER)
    touched = {
        index // 3 % 8
        for index in range(0, len(canvas.pixels), 3)
        if canvas.pixels[index : index + 3] != before[index : index + 3]
    }
    # Anything that landed did so inside the requested column span.
    assert all(x <= column < x + w for column in touched)


def test_blitting_past_the_left_edge_does_not_wrap_onto_the_row_above():
    """The failure this guards against looks like a rendering artefact, not a bug."""
    canvas = render.Raster(10, 3)
    patch = render.Raster(4, 1, fill=render.ALERT)
    canvas.blit(patch, -2, 1)
    assert [pixel(canvas, x, 1) for x in range(3)] == [render.ALERT, render.ALERT, render.GROUND]
    assert all(pixel(canvas, x, 0) == render.GROUND for x in range(10))


def test_blitting_past_the_right_edge_stays_on_its_own_row():
    canvas = render.Raster(6, 2)
    canvas.blit(render.Raster(4, 1, fill=render.GOOD), 4, 0)
    assert [pixel(canvas, x, 0) for x in (4, 5)] == [render.GOOD, render.GOOD]
    assert all(pixel(canvas, x, 1) == render.GROUND for x in range(6))


def test_text_advances_by_the_width_it_reports():
    canvas = render.Raster(120, 12)
    end = canvas.text(3, 2, "AB", render.PAPER)
    assert end - 3 == render.text_width("AB")
    assert any(pixel(canvas, x, 3) == render.PAPER for x in range(3, end))


def test_an_unknown_character_renders_as_a_question_mark_rather_than_nothing():
    known = render.Raster(40, 10)
    known.text(0, 0, "?", render.PAPER)
    unknown = render.Raster(40, 10)
    unknown.text(0, 0, "é", render.PAPER)
    assert unknown.pixels == known.pixels


def test_the_font_covers_every_character_the_captions_use():
    for glyph in render.FONT.values():
        assert len(glyph) == render.GLYPH_HEIGHT
        assert all(len(row) == render.GLYPH_WIDTH for row in glyph)
    # Captions are upper-cased on the way in, so lowercase never needs a glyph.
    assert not any(char.islower() for char in render.FONT)


# --- blueprints ------------------------------------------------------------


def test_the_board_is_the_blueprint_extent_plus_padding():
    blueprint, _ = synth.GENERATORS["smelter-column"](random.Random(1), DATA)
    tiles_w, tiles_h = blueprint.normalised(DATA).extent(DATA)
    canvas = render.board(blueprint, DATA, scale=10, pad=1)
    assert (canvas.width, canvas.height) == ((tiles_w + 2) * 10, (tiles_h + 2) * 10)


def test_an_entity_is_drawn_in_its_own_colour_at_its_own_tile():
    belt = DATA.entities["transport-belt"]
    blueprint = Blueprint(entities=[Placement("transport-belt", 3, 2, EAST)]).normalised(DATA)
    canvas = render.board(blueprint, DATA, scale=10, pad=1, grid=False)
    # Normalisation moves it to the origin, so it sits at the padding offset.
    assert pixel(canvas, 15, 15) == render.shade(render.ink(belt.color), 0.7)
    assert pixel(canvas, 5, 5) == render.GROUND


def test_a_multi_tile_machine_covers_all_of_its_tiles():
    blueprint = Blueprint(
        entities=[Placement("assembling-machine-2", 0, 0, NORTH, recipe="iron-gear-wheel")]
    ).normalised(DATA)
    canvas = render.board(blueprint, DATA, scale=10, pad=0, grid=False)
    assert (canvas.width, canvas.height) == (30, 30)
    # A 3x3 with a recipe: body colour at the corners, the recipe pip in the middle.
    assert pixel(canvas, 25, 25) != render.GROUND
    assert pixel(canvas, 15, 15) == render.PAPER


@pytest.mark.parametrize(
    "direction,probe",
    [(NORTH, (5, 0)), (SOUTH, (5, 9)), (EAST, (9, 5)), (WEST, (0, 5))],
)
def test_the_facing_bar_sits_on_the_leading_edge(direction, probe):
    """Which way a belt points is the single most load-bearing pixel here."""
    blueprint = Blueprint(entities=[Placement("transport-belt", 0, 0, direction)]).normalised(DATA)
    canvas = render.board(blueprint, DATA, scale=10, pad=0, grid=False)
    assert pixel(canvas, *probe) == render.PAPER


def test_a_tunnel_entrance_and_its_exit_are_told_apart():
    def middle(flow):
        blueprint = Blueprint(
            entities=[Placement("underground-belt", 0, 0, EAST, flow=flow)]
        ).normalised(DATA)
        return pixel(render.board(blueprint, DATA, scale=12, pad=0, grid=False), 6, 6)

    assert middle("input") == render.PAPER  # solid pip
    assert middle("output") != render.PAPER  # hollow


def test_an_unknown_prototype_is_skipped_rather_than_crashing():
    """Model output will name entities that do not exist. Draw the rest anyway."""
    blueprint = Blueprint(
        entities=[
            Placement("transport-belt", 0, 0, EAST),
            Placement("nonsense-machine", 2, 0, EAST),
        ]
    )
    canvas = render.board(blueprint, DATA, scale=8, pad=0, grid=False)
    assert (canvas.width, canvas.height) == (8, 8)  # sized to the belt alone
    assert any(pixel(canvas, x, 4) != render.GROUND for x in range(canvas.width))
    assert ElementTree.fromstring(render.svg(blueprint, DATA)) is not None


def test_an_empty_blueprint_still_renders_a_plate():
    canvas = render.board(Blueprint(entities=[]), DATA, scale=8)
    assert (canvas.width, canvas.height) == (24, 24)


# --- cards and sheets ------------------------------------------------------


def test_every_card_is_the_same_size_whatever_it_contains():
    """A ragged grid reads size as importance; these are the same object."""
    small = Blueprint(entities=[Placement("transport-belt", 0, 0, EAST)]).normalised(DATA)
    large, _ = synth.GENERATORS["bus-tap"](random.Random(0), DATA)
    cards = [
        render.card(small, DATA, title="SMALL", lines=("A", "B")),
        render.card(large, DATA, title="LARGE", lines=("A", "B")),
    ]
    assert cards[0].width == cards[1].width
    assert cards[0].height == cards[1].height


def test_a_card_keeps_its_drawing_inside_its_own_frame():
    blueprint, _ = synth.GENERATORS["balancer"](random.Random(2), DATA)
    canvas = render.card(blueprint, DATA, title="BALANCER", lines=("VALID",), accent=render.GOOD)
    edge = render.shade(render.GOOD, 0.35)
    assert pixel(canvas, 0, 0) == edge
    assert pixel(canvas, canvas.width - 1, canvas.height - 1) == edge


def test_a_caption_too_long_for_the_plate_is_truncated_not_overrun():
    blueprint = Blueprint(entities=[Placement("transport-belt", 0, 0, EAST)]).normalised(DATA)
    canvas = render.card(blueprint, DATA, box=(60, 40), title="X" * 200, lines=("Y" * 200,))
    # The frame on the right edge survives, so nothing was drawn through it.
    assert pixel(canvas, canvas.width - 1, 10) == render.shade(render.PAPER, 0.35)


def test_a_sheet_lays_cards_out_on_a_regular_grid():
    cards = [render.Raster(20, 10, fill=render.GOOD) for _ in range(5)]
    canvas = render.sheet(cards, columns=2, gap=4)
    assert (canvas.width, canvas.height) == (2 * 20 + 3 * 4, 3 * 10 + 4 * 4)
    assert pixel(canvas, 4, 4) == render.GOOD  # first cell
    assert pixel(canvas, 4 + 20 + 4, 4) == render.GOOD  # second column
    assert pixel(canvas, 4 + 20, 4) == render.GROUND  # the gap between them


def test_a_sheet_of_nothing_is_not_a_crash():
    assert render.sheet([]).png()


# --- measurements ----------------------------------------------------------


def test_occupancy_counts_every_tile_of_every_entity():
    blueprint = Blueprint(
        entities=[
            Placement("assembling-machine-1", 0, 0, NORTH),  # 3x3
            Placement("transport-belt", 4, 0, EAST),  # 1x1
        ]
    ).normalised(DATA)
    counts = render.occupancy([blueprint, blueprint], DATA)
    assert len(counts) == len(counts[0]) == GRID
    assert sum(sum(row) for row in counts) == 2 * (9 + 1)
    assert counts[0][0] == 2
    assert counts[GRID - 1][GRID - 1] == 0


def test_occupancy_ignores_an_entity_it_cannot_place():
    counts = render.occupancy([Blueprint(entities=[Placement("nonsense-machine", 0, 0)])], DATA)
    assert sum(sum(row) for row in counts) == 0


def test_trim_crops_to_what_was_built_on():
    counts = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 3, 0], [0, 0, 0, 0]]
    assert render.trim(counts) == [[1, 2], [0, 3]]


def test_trim_leaves_an_empty_grid_alone_rather_than_returning_nothing():
    # A heatmap over zero designs still has to draw: a raster of no width is a
    # crash, and an empty square says "nothing here" perfectly well.
    assert render.trim([[0, 0], [0, 0]]) == [[0, 0], [0, 0]]


def test_the_heatmap_bands_are_monotone_in_the_count():
    counts = [[0, 1, 4], [8, 16, 32]]
    canvas = render.heatmap(counts, scale=4, title="T")
    bands = [
        render.HEAT.index(pixel(canvas, 10 + x * 4 + 1, 22 + y * 4 + 1))
        for y in (0, 1)
        for x in (0, 1, 2)
    ]
    assert bands == sorted(bands)
    assert bands[0] == 0  # a zero tile stays ground-coloured
    assert bands[-1] == len(render.HEAT) - 1  # the peak reaches the top of the ramp


def test_an_all_zero_heatmap_does_not_divide_by_zero():
    canvas = render.heatmap([[0, 0], [0, 0]], scale=4)
    assert canvas.png()


# --- vectors ---------------------------------------------------------------


def test_the_svg_is_parseable_xml_with_the_declared_size():
    blueprint, _ = synth.GENERATORS["mall-cell"](random.Random(3), DATA)
    document = render.svg(blueprint, DATA, scale=16, title="MALL", lines=("VALID",))
    root = ElementTree.fromstring(document)
    assert root.tag.endswith("svg")
    tiles_w, _ = blueprint.normalised(DATA).extent(DATA)
    assert root.attrib["width"] == str((tiles_w + 2) * 16)
    assert root.attrib["viewBox"] == f"0 0 {root.attrib['width']} {root.attrib['height']}"
    assert root.attrib["shape-rendering"] == "crispEdges"


def test_the_svg_draws_one_rectangle_per_entity_at_least():
    blueprint = Blueprint(
        entities=[Placement("assembling-machine-1", 0, 0, NORTH, recipe="electronic-circuit")]
    ).normalised(DATA)
    document = render.svg(blueprint, DATA, scale=16, pad=0)
    root = ElementTree.fromstring(document)
    rectangles = [node for node in root.iter() if node.tag.endswith("rect")]
    machine = DATA.entities["assembling-machine-1"]
    assert any(node.attrib.get("stroke") == machine.color for node in rectangles)
    # The recipe, abbreviated to fit a 3x3. Lowercase in the markup because the
    # stylesheet upper-cases it at draw time, so the text stays selectable as
    # the real recipe name.
    texts = [node.text for node in root.iter() if node.tag.endswith("text")]
    assert "el-ci" in texts


def test_markup_in_a_caption_cannot_break_the_document():
    blueprint = Blueprint(entities=[Placement("transport-belt", 0, 0, EAST)]).normalised(DATA)
    document = render.svg(blueprint, DATA, title="<bp> & </bp>", lines=("A < B",))
    root = ElementTree.fromstring(document)
    assert "<bp> & </bp>" in [node.text for node in root.iter() if node.tag.endswith("text")]


@pytest.mark.parametrize("kind", sorted(synth.GENERATORS))
def test_every_generator_renders_in_both_formats(kind):
    """A generator whose output cannot be looked at is a generator nobody trusts."""
    blueprint, spec = synth.GENERATORS[kind](random.Random(7), DATA)
    assert render.card(blueprint, DATA, title=kind, lines=(f"{spec.width}X{spec.height}",)).png()
    assert ElementTree.fromstring(render.svg(blueprint, DATA, title=kind)) is not None
