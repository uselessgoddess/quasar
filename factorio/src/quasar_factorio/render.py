"""Drawing blueprints without a single sprite.

Factorio's own textures are 300 MB of copyrighted PNGs, and a texture atlas is
the wrong tool for the question this harness asks anyway. What matters when you
look at a generated factory is: how big is it, what is where, which way is it
facing, is it a solid block or a scatter of debris. All four survive perfectly
in flat colour, and flat colour renders in a millisecond with no dependency, so
a training loop can emit a frame per checkpoint and end up with a timelapse.

The style is deliberate: near-black ground, one tile grid line, full-bleed
colour blocks with hard edges, a white bar on the facing side, and a 5x7 bitmap
font. No gradients, no shadows, no rounded corners, no anti-aliasing. Brutalist
because the alternative — softening everything — hides exactly the artefacts
worth seeing, like a row of belts that is one tile out of line.

Two outputs, same drawing code:

* :func:`svg` for a single blueprint you want to zoom into, with real text.
* :class:`Raster` / `.png()` for anything that has to be piped into a video or
  laid out in a contact sheet.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Iterable, Sequence

from .blueprint import GRID, Blueprint
from .prototypes import EAST, NORTH, SOUTH, WEST, Data, load

RGB = tuple[int, int, int]

# The whole palette. Everything else on screen is an entity colour from the
# prototype table, which is why there are so few entries here.
GROUND: RGB = (10, 10, 11)
RULE: RGB = (26, 26, 28)
CHUNK: RGB = (38, 38, 42)
PAPER: RGB = (238, 238, 238)
FAINT: RGB = (120, 120, 126)
ALERT: RGB = (226, 74, 58)
GOOD: RGB = (110, 200, 120)

# Occupancy ramp, dark to hot. Read as a step function, not interpolated: a
# smooth ramp reads as a photograph, and this is a measurement.
HEAT: tuple[RGB, ...] = (
    (10, 10, 11),
    (28, 30, 56),
    (46, 62, 110),
    (44, 116, 140),
    (86, 168, 110),
    (196, 182, 64),
    (232, 128, 48),
    (238, 238, 238),
)


def ink(color: str) -> RGB:
    """`#rrggbb` -> bytes."""
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def shade(color: RGB, factor: float) -> RGB:
    return tuple(max(0, min(255, round(channel * factor))) for channel in color)  # type: ignore[return-value]


_FONT_ROWS = {
    "A": "01110 10001 10001 11111 10001 10001 10001",
    "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110",
    "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111",
    "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111",
    "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111",
    "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001",
    "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001",
    "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110",
    "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101",
    "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110",
    "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110",
    "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 11011 10001",
    "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100",
    "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110",
    "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111",
    "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010",
    "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110",
    "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110",
    "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000",
    "-": "00000 00000 00000 11111 00000 00000 00000",
    "_": "00000 00000 00000 00000 00000 00000 11111",
    ".": "00000 00000 00000 00000 00000 01100 01100",
    ",": "00000 00000 00000 00000 01100 01100 01000",
    ":": "00000 01100 01100 00000 01100 01100 00000",
    "/": "00001 00010 00010 00100 01000 01000 10000",
    "%": "11001 11010 00010 00100 01000 01011 10011",
    "+": "00000 00100 00100 11111 00100 00100 00000",
    "=": "00000 00000 11111 00000 11111 00000 00000",
    "<": "00010 00100 01000 10000 01000 00100 00010",
    ">": "01000 00100 00010 00001 00010 00100 01000",
    "#": "01010 01010 11111 01010 11111 01010 01010",
    "(": "00010 00100 01000 01000 01000 00100 00010",
    ")": "01000 00100 00010 00010 00010 00100 01000",
    "!": "00100 00100 00100 00100 00100 00000 00100",
    "?": "01110 10001 00001 00010 00100 00000 00100",
    "*": "00000 10101 01110 11111 01110 10101 00000",
    "[": "01110 01000 01000 01000 01000 01000 01110",
    "]": "01110 00010 00010 00010 00010 00010 01110",
}
GLYPH_WIDTH, GLYPH_HEIGHT, GLYPH_GAP = 5, 7, 1
FONT = {char: rows.split() for char, rows in _FONT_ROWS.items()}


def text_width(message: str, scale: int = 1) -> int:
    return len(message) * (GLYPH_WIDTH + GLYPH_GAP) * scale


class Raster:
    """A flat RGB framebuffer with the six primitives this module needs."""

    def __init__(self, width: int, height: int, fill: RGB = GROUND) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self.pixels = bytearray(bytes(fill) * (self.width * self.height))

    def dot(self, x: int, y: int, color: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            at = (y * self.width + x) * 3
            self.pixels[at : at + 3] = bytes(color)

    def box(self, x: int, y: int, width: int, height: int, color: RGB) -> None:
        """A filled rectangle, clipped, written a row at a time."""
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + width), min(self.height, y + height)
        if x1 <= x0 or y1 <= y0:
            return
        row = bytes(color) * (x1 - x0)
        for line in range(y0, y1):
            at = (line * self.width + x0) * 3
            self.pixels[at : at + len(row)] = row

    def frame(self, x: int, y: int, width: int, height: int, color: RGB, weight: int = 1) -> None:
        self.box(x, y, width, weight, color)
        self.box(x, y + height - weight, width, weight, color)
        self.box(x, y, weight, height, color)
        self.box(x + width - weight, y, weight, height, color)

    def text(self, x: int, y: int, message: str, color: RGB, scale: int = 1) -> int:
        """Uppercase 5x7. Returns the x the next string would start at."""
        cursor = x
        for char in message.upper():
            glyph = FONT.get(char, FONT["?"])
            for row, bits in enumerate(glyph):
                for column, bit in enumerate(bits):
                    if bit == "1":
                        self.box(cursor + column * scale, y + row * scale, scale, scale, color)
            cursor += (GLYPH_WIDTH + GLYPH_GAP) * scale
        return cursor

    def blit(self, other: Raster, x: int, y: int) -> None:
        """Copy `other` in at `(x, y)`, clipped on all four sides.

        Clipping the columns matters as much as the rows: a row copied past the
        right edge would reappear at the start of the next one, and the result
        looks like a rendering artefact rather than a bounds bug.
        """
        left, right = max(0, -x), min(other.width, self.width - x)
        if right <= left:
            return
        span = (right - left) * 3
        for line in range(max(0, -y), min(other.height, self.height - y)):
            source = (line * other.width + left) * 3
            at = ((y + line) * self.width + x + left) * 3
            self.pixels[at : at + span] = other.pixels[source : source + span]

    def png(self) -> bytes:
        """PNG bytes. Filter 0 on every row; zlib does the rest."""
        raw = bytearray()
        stride = self.width * 3
        for line in range(self.height):
            raw.append(0)
            raw += self.pixels[line * stride : (line + 1) * stride]
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                _chunk(b"IHDR", header),
                _chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
                _chunk(b"IEND", b""),
            ]
        )


def _chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def drawable(blueprint: Blueprint, data: Data) -> Blueprint:
    """The blueprint with anything the prototype table does not know dropped.

    Model output is the input here, and a sampler will happily invent
    `assembling-machine-4`. Every geometric operation — `bounds`, `normalised`,
    `tiles` — raises on a name it cannot size, so an unfiltered blueprint makes
    the renderer throw exactly when there is something interesting to look at.
    Dropping the unknown entities draws the rest; the grader is what reports
    that they were there.
    """
    known = [entity for entity in blueprint.entities if data.entity(entity.name) is not None]
    if len(known) == len(blueprint.entities):
        return blueprint
    return Blueprint(label=blueprint.label, source=blueprint.source, entities=known)


def board(
    blueprint: Blueprint,
    data: Data | None = None,
    *,
    scale: int = 10,
    pad: int = 1,
    grid: bool = True,
) -> Raster:
    """The blueprint itself, one rectangle per entity, `scale` pixels per tile."""
    data = data or load()
    blueprint = drawable(blueprint, data).normalised(data)
    width, height = blueprint.extent(data)
    width, height = max(1, width) + 2 * pad, max(1, height) + 2 * pad
    canvas = Raster(width * scale, height * scale)

    if grid and scale >= 4:
        for column in range(width + 1):
            canvas.box(column * scale, 0, 1, canvas.height, CHUNK if column % 8 == 0 else RULE)
        for row in range(height + 1):
            canvas.box(0, row * scale, canvas.width, 1, CHUNK if row % 8 == 0 else RULE)

    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        tiles_w, tiles_h = proto.footprint(placement.direction)
        x = (placement.x + pad) * scale
        y = (placement.y + pad) * scale
        w, h = tiles_w * scale, tiles_h * scale
        color = ink(proto.color)
        canvas.box(x, y, w, h, shade(color, 0.7))
        canvas.frame(x, y, w, h, color, weight=max(1, scale // 8))
        _facing(canvas, placement.direction, x, y, w, h, scale)
        if proto.takes_flow:
            _flow(canvas, placement.flow, x, y, w, h)
        if proto.takes_recipe and placement.recipe:
            # A single bright pip: the recipe is in the caption, and a legible
            # name would need more pixels than a 3x3 machine has.
            canvas.box(x + w // 2 - 1, y + h // 2 - 1, 3, 3, PAPER)
    return canvas


def _facing(canvas: Raster, direction: int, x: int, y: int, w: int, h: int, scale: int) -> None:
    """A pale bar along the edge the entity faces."""
    bar = max(1, scale // 4)
    if direction == NORTH:
        canvas.box(x, y, w, bar, PAPER)
    elif direction == SOUTH:
        canvas.box(x, y + h - bar, w, bar, PAPER)
    elif direction == EAST:
        canvas.box(x + w - bar, y, bar, h, PAPER)
    elif direction == WEST:
        canvas.box(x, y, bar, h, PAPER)
    else:  # the four rail-only diagonals: mark the corner instead.
        corner_x = x if direction in (5, 7) else x + w - bar
        corner_y = y if direction in (1, 7) else y + h - bar
        canvas.box(corner_x, corner_y, bar, bar, PAPER)


def _flow(canvas: Raster, flow: str | None, x: int, y: int, w: int, h: int) -> None:
    """Solid pip for a tunnel entrance, hollow for its exit."""
    size = max(2, min(w, h) // 2)
    cx, cy = x + (w - size) // 2, y + (h - size) // 2
    if flow == "output":
        canvas.frame(cx, cy, size, size, PAPER)
    else:
        canvas.box(cx, cy, size, size, PAPER)


def card(
    blueprint: Blueprint,
    data: Data | None = None,
    *,
    box: tuple[int, int] = (248, 168),
    title: str = "",
    lines: Sequence[str] = (),
    accent: RGB = PAPER,
    pad: int = 1,
) -> Raster:
    """A blueprint with a heading and caption lines, in a fixed-size plate.

    Fixed size, with the drawing scaled to fit and centred, is what makes a
    contact sheet legible: a 3x7 mall cell and a 40x12 bus tap are the same
    object to the reader, and letting them set their own dimensions produces a
    ragged grid where size reads as importance.

    The caption carries the grader's verdict, so a sheet needs no legend — the
    rule under the title is green or red, and the lines say why.
    """
    data = data or load()
    tiles_w, tiles_h = drawable(blueprint, data).extent(data)
    scale = max(
        2, min(box[0] // (max(1, tiles_w) + 2 * pad), box[1] // (max(1, tiles_h) + 2 * pad))
    )
    plate = board(blueprint, data, scale=min(scale, 24), pad=pad)

    margin = 8
    head = GLYPH_HEIGHT + 6 if title else 0
    canvas = Raster(
        box[0] + 2 * margin,
        head + box[1] + (GLYPH_HEIGHT + 3) * len(lines) + 2 * margin + (6 if lines else 0),
    )
    if title:
        canvas.text(margin, margin, _clip(title, box[0]), accent, 1)
        canvas.box(margin, margin + GLYPH_HEIGHT + 2, box[0], 1, accent)
    canvas.blit(
        plate,
        margin + (box[0] - plate.width) // 2,
        margin + head + (box[1] - plate.height) // 2,
    )
    y = margin + head + box[1] + 6
    for line in lines:
        canvas.text(margin, y, _clip(line, box[0]), FAINT, 1)
        y += GLYPH_HEIGHT + 3
    canvas.frame(0, 0, canvas.width, canvas.height, shade(accent, 0.35))
    return canvas


def _clip(message: str, width: int) -> str:
    """Truncate to what fits, rather than letting it run off the plate."""
    room = max(1, width // (GLYPH_WIDTH + GLYPH_GAP))
    return message if len(message) <= room else message[: room - 1] + "*"


def sheet(cards: Sequence[Raster], columns: int = 4, gap: int = 8) -> Raster:
    """Lay cards out in a grid. One frame of a timelapse, or one eval batch."""
    if not cards:
        return Raster(1, 1)
    columns = max(1, min(columns, len(cards)))
    rows = (len(cards) + columns - 1) // columns
    cell_w = max(card.width for card in cards)
    cell_h = max(card.height for card in cards)
    canvas = Raster(columns * cell_w + (columns + 1) * gap, rows * cell_h + (rows + 1) * gap)
    for index, item in enumerate(cards):
        column, row = index % columns, index // columns
        canvas.blit(item, gap + column * (cell_w + gap), gap + row * (cell_h + gap))
    return canvas


def occupancy(blueprints: Iterable[Blueprint], data: Data | None = None) -> list[list[int]]:
    """How often each tile of the grid is occupied across many blueprints."""
    data = data or load()
    counts = [[0] * GRID for _ in range(GRID)]
    for blueprint in blueprints:
        for placement in drawable(blueprint, data).normalised(data).entities:
            for x, y in placement.tiles(data):
                if 0 <= x < GRID and 0 <= y < GRID:
                    counts[y][x] += 1
    return counts


def trim(counts: Sequence[Sequence[int]]) -> list[list[int]]:
    """The occupied part of an occupancy grid, without the empty margin.

    Designs are anchored at the origin and almost all of them are far smaller
    than the 64x64 grid they are placed on, so a full-grid heatmap spends three
    quarters of its pixels on ground nothing was ever built on. Cropping to the
    bounding box of the non-zero tiles keeps the shape and the aspect ratio and
    puts the pixels where the signal is. An empty grid is returned unchanged —
    there is no bounding box to crop to, and a zero-sized raster is worse than
    an honest empty one.
    """
    rows = [y for y, row in enumerate(counts) if any(row)]
    if not rows:
        return [list(row) for row in counts]
    width = max(len(row) for row in counts)
    columns = [x for x in range(width) if any(x < len(row) and row[x] for row in counts)]
    return [list(row[columns[0] : columns[-1] + 1]) for row in counts[rows[0] : rows[-1] + 1]]


def heatmap(
    counts: Sequence[Sequence[int]],
    *,
    scale: int = 8,
    title: str = "OCCUPANCY",
) -> Raster:
    """The occupancy grid as a stepped heat plot with a legend strip.

    Steps rather than a smooth ramp: eight bands can be counted off against the
    legend, and a continuous gradient invites reading precision that a count of
    a few hundred samples does not have.
    """
    rows, columns = len(counts), max((len(row) for row in counts), default=0)
    peak = max((max(row, default=0) for row in counts), default=0) or 1
    margin, legend = 10, 14
    canvas = Raster(columns * scale + 2 * margin, rows * scale + 2 * margin + legend + 12)
    canvas.text(margin, margin - 8 + 2, f"{title}  PEAK {peak}", PAPER, 1)
    top = margin + 12
    for y, row in enumerate(counts):
        for x, value in enumerate(row):
            band = 0 if value == 0 else 1 + int((len(HEAT) - 2) * value / peak)
            canvas.box(
                margin + x * scale, top + y * scale, scale, scale, HEAT[min(band, len(HEAT) - 1)]
            )
    canvas.frame(margin - 1, top - 1, columns * scale + 2, rows * scale + 2, FAINT)
    strip = top + rows * scale + 6
    step = max(1, (columns * scale) // len(HEAT))
    for index, color in enumerate(HEAT):
        canvas.box(margin + index * step, strip, step, 6, color)
    canvas.text(margin, strip + 8, f"0{' ' * 6}{peak}", FAINT, 1)
    return canvas


_SVG_HEAD = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
    'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">'
)


def svg(
    blueprint: Blueprint,
    data: Data | None = None,
    *,
    scale: int = 16,
    pad: int = 1,
    title: str = "",
    lines: Sequence[str] = (),
) -> str:
    """The same drawing as vectors, with real text and recipe labels.

    Worth having separately from the raster path: an SVG stays sharp at any zoom
    and can carry the recipe name inside the machine that runs it, which is the
    one piece of information the pixel renderer has to leave out.
    """
    data = data or load()
    blueprint = drawable(blueprint, data).normalised(data)
    width, height = blueprint.extent(data)
    width, height = (max(1, width) + 2 * pad) * scale, (max(1, height) + 2 * pad) * scale
    head = 26 if title else 0
    foot = 16 * len(lines)
    parts = [_SVG_HEAD.format(w=width, h=height + head + foot)]
    parts.append(f'<rect width="100%" height="100%" fill="{_hex(GROUND)}"/>')
    parts.append(
        '<style>text{font-family:"IBM Plex Mono","DejaVu Sans Mono",monospace;'
        "text-transform:uppercase;letter-spacing:.06em}</style>"
    )
    if title:
        parts.append(
            f'<text x="6" y="16" fill="{_hex(PAPER)}" font-size="12">{_escape(title)}</text>'
        )
        parts.append(f'<rect x="6" y="20" width="{width - 12}" height="1" fill="{_hex(PAPER)}"/>')

    for column in range(width // scale + 1):
        colour = _hex(CHUNK if column % 8 == 0 else RULE)
        parts.append(
            f'<rect x="{column * scale}" y="{head}" width="1" height="{height}" fill="{colour}"/>'
        )
    for row in range(height // scale + 1):
        colour = _hex(CHUNK if row % 8 == 0 else RULE)
        parts.append(
            f'<rect x="0" y="{head + row * scale}" width="{width}" height="1" fill="{colour}"/>'
        )

    for placement in blueprint.entities:
        proto = data.entity(placement.name)
        tiles_w, tiles_h = proto.footprint(placement.direction)
        x, y = (placement.x + pad) * scale, head + (placement.y + pad) * scale
        w, h = tiles_w * scale, tiles_h * scale
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{_dim(proto.color)}" '
            f'stroke="{proto.color}" stroke-width="{max(1, scale // 8)}"/>'
        )
        parts.append(_svg_facing(placement.direction, x, y, w, h, scale))
        if proto.takes_flow:
            fill = "none" if placement.flow == "output" else _hex(PAPER)
            size = max(2, min(w, h) // 2)
            parts.append(
                f'<rect x="{x + (w - size) // 2}" y="{y + (h - size) // 2}" width="{size}" '
                f'height="{size}" fill="{fill}" stroke="{_hex(PAPER)}"/>'
            )
        if placement.recipe and w >= 3 * scale:
            parts.append(
                f'<text x="{x + w // 2}" y="{y + h // 2 + 3}" fill="{_hex(PAPER)}" font-size="7" '
                f'text-anchor="middle">{_escape(_abbreviate(placement.recipe))}</text>'
            )

    for index, line in enumerate(lines):
        parts.append(
            f'<text x="6" y="{head + height + 12 + index * 16}" fill="{_hex(FAINT)}" '
            f'font-size="10">{_escape(line)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _svg_facing(direction: int, x: int, y: int, w: int, h: int, scale: int) -> str:
    bar = max(1, scale // 4)
    spans = {
        NORTH: (x, y, w, bar),
        SOUTH: (x, y + h - bar, w, bar),
        EAST: (x + w - bar, y, bar, h),
        WEST: (x, y, bar, h),
    }
    left, top, width, height = spans.get(direction, (x, y, bar, bar))
    return f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="{_hex(PAPER)}"/>'


def _abbreviate(name: str) -> str:
    """`electronic-circuit` -> `EL-CI`: enough to tell recipes apart in a 3x3."""
    return "-".join(word[:2] for word in name.split("-")[:2])


def _hex(color: RGB) -> str:
    return "#{:02x}{:02x}{:02x}".format(*color)


def _dim(color: str) -> str:
    return _hex(shade(ink(color), 0.7))


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
