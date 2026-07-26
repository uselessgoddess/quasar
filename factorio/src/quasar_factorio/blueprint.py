"""Blueprint strings in, tile-grid blueprints out.

A Factorio blueprint string is `"0"` followed by base64 of zlib-compressed JSON.
Inside, positions are the *centre* of an entity in tiles, which lands on half
tiles for odd-sized buildings. The model is far better served by integer
top-left corners on a small grid, so that conversion — and its exact inverse —
lives here and nowhere else.
"""

from __future__ import annotations

import base64
import json
import math
import zlib
from dataclasses import dataclass, field, replace

from .prototypes import EAST, NORTH, SOUTH, WEST, Data, load

# The grid the model writes on. 64x64 covers a smelting column, a mall cell or a
# science block with room to spare, and keeps the coordinate vocabulary at 128
# tokens. Anything bigger is a different problem than the one this harness poses.
GRID = 64


class BlueprintError(ValueError):
    """A blueprint that cannot be represented on the harness grid."""


@dataclass(frozen=True)
class Placement:
    """One entity, pinned to its top-left tile."""

    name: str
    x: int
    y: int
    direction: int = NORTH
    recipe: str | None = None
    #: `"input"` or `"output"` for underground belts, `None` for everything else.
    flow: str | None = None

    def size(self, data: Data) -> tuple[int, int]:
        proto = data.entity(self.name)
        if proto is None:
            raise BlueprintError(f"unknown entity {self.name!r}")
        return proto.footprint(self.direction)

    def tiles(self, data: Data):
        width, height = self.size(data)
        for dy in range(height):
            for dx in range(width):
                yield self.x + dx, self.y + dy


@dataclass
class Blueprint:
    """A normalised blueprint: integer tiles, origin at (0, 0), raster order."""

    entities: list[Placement] = field(default_factory=list)
    label: str = ""
    source: str = ""

    def bounds(self, data: Data) -> tuple[int, int, int, int]:
        if not self.entities:
            return 0, 0, 0, 0
        xs, ys = [], []
        for placement in self.entities:
            width, height = placement.size(data)
            xs += [placement.x, placement.x + width]
            ys += [placement.y, placement.y + height]
        return min(xs), min(ys), max(xs), max(ys)

    def extent(self, data: Data) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bounds(data)
        return x1 - x0, y1 - y0

    def normalised(self, data: Data) -> Blueprint:
        """Shift to the origin and sort into raster order.

        Canonical order matters more than it looks: an autoregressive model that
        sees the same layout arrive in a different sequence each epoch has to
        spend capacity learning that the order is arbitrary. Raster order (row,
        then column, then name) makes the next token a local question.
        """
        x0, y0, _, _ = self.bounds(data)
        moved = [replace(e, x=e.x - x0, y=e.y - y0) for e in self.entities]
        moved.sort(key=lambda e: (e.y, e.x, e.name, e.direction))
        return Blueprint(entities=moved, label=self.label, source=self.source)

    def fits(self, data: Data, grid: int = GRID) -> bool:
        width, height = self.extent(data)
        return width <= grid and height <= grid


def decode_string(text: str) -> dict:
    """`0eNq...` -> the blueprint JSON."""
    text = text.strip()
    if not text or text[0] != "0":
        raise BlueprintError("not a version-0 blueprint string")
    try:
        return json.loads(zlib.decompress(base64.b64decode(text[1:])))
    except Exception as error:  # noqa: BLE001 - the library errors are all opaque
        raise BlueprintError(f"undecodable blueprint string: {error}") from error


def encode_string(payload: dict) -> str:
    """The inverse of :func:`decode_string`, at zlib's maximum level.

    Factorio itself uses level 9; matching it keeps strings byte-identical to
    what the game would produce for the same JSON, which makes diffing samples
    against game output meaningful.
    """
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return "0" + base64.b64encode(zlib.compress(raw, 9)).decode()


def books(payload: dict):
    """Yield every `blueprint` object in a blueprint or blueprint book.

    Books nest, and about a third of what factorioprints hosts is a book, so
    walking them is the difference between 12k usable blueprints and 8k.
    """
    if "blueprint" in payload:
        yield payload["blueprint"]
    for entry in payload.get("blueprint_book", {}).get("blueprints", []) or []:
        yield from books(entry)


def from_json(payload: dict, data: Data | None = None, *, strict: bool = False) -> Blueprint:
    """Convert one `blueprint` object into a normalised :class:`Blueprint`.

    Unknown prototypes (mods, 2.0-only entities, tiles) are skipped unless
    `strict`, because a single modded lamp should not cost us an otherwise
    perfect smelter design.
    """
    data = data or load()
    placements = []
    for entity in payload.get("entities", []) or []:
        name = entity.get("name")
        proto = data.entity(name)
        if proto is None:
            if strict:
                raise BlueprintError(f"unknown entity {name!r}")
            continue
        direction = int(entity.get("direction", NORTH)) % 8
        width, height = proto.footprint(direction)
        position = entity.get("position") or {}
        try:
            centre_x = float(position["x"])
            centre_y = float(position["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise BlueprintError(f"entity {name!r} has no usable position") from error
        placements.append(
            Placement(
                name=name,
                x=_tile(centre_x - width / 2),
                y=_tile(centre_y - height / 2),
                direction=direction,
                recipe=entity.get("recipe") if proto.takes_recipe else None,
                flow=_flow(entity.get("type")) if proto.takes_flow else None,
            )
        )
    label = payload.get("label") or ""
    return Blueprint(entities=placements, label=label).normalised(data)


def to_json(blueprint: Blueprint, data: Data | None = None, *, label: str = "") -> dict:
    """Convert back to the JSON the game accepts.

    Entity numbers are 1-based and dense; the game tolerates gaps but its own
    exporter never emits them, and a few third-party parsers assume it.
    """
    data = data or load()
    entities = []
    for number, placement in enumerate(blueprint.entities, start=1):
        width, height = placement.size(data)
        entity = {
            "entity_number": number,
            "name": placement.name,
            "position": {"x": placement.x + width / 2, "y": placement.y + height / 2},
        }
        if placement.direction:
            entity["direction"] = placement.direction
        if placement.recipe:
            entity["recipe"] = placement.recipe
        if data.entity(placement.name) is not None and data.entities[placement.name].takes_flow:
            entity["type"] = placement.flow or "input"
        entities.append(entity)
    return {
        "blueprint": {
            "item": "blueprint",
            "label": label or blueprint.label or "quasar",
            "version": 281479273775104,
            "icons": icons(blueprint, data),
            "entities": entities,
        }
    }


def _tile(corner: float) -> int:
    """A corner in tiles, rounded half *up* rather than to even.

    Not a nitpick: `round` in Python breaks ties towards the even number, so
    -1.5 and -2.5 both become -2. Blueprints written by Factorio 0.15 carry a
    half-tile offset on the whole design — a belt sits at an integer position
    where a modern export would put it at x.5 — and under banker's rounding
    every second row of such a blueprint collapses onto the one before it. That
    turns a perfectly good balancer into a pile of overlapping belts, which the
    grader then rejects; it accounted for 462 of the first 465 rejections when
    this was measured against factorioprints.

    Half-up keeps the whole design shifted by one tile instead, which
    `normalised` immediately shifts back.
    """
    return math.floor(corner + 0.5)


def _flow(value: object) -> str:
    """Normalise the underground-belt `type` field, defaulting to the entrance."""
    return "output" if str(value) == "output" else "input"


def icons(blueprint: Blueprint, data: Data) -> list[dict]:
    """Up to four icons, taken from the most common entities.

    Without icons the game shows a blank blueprint, which makes a folder of
    samples unreadable in the inventory.
    """
    counts: dict[str, int] = {}
    for placement in blueprint.entities:
        counts[placement.name] = counts.get(placement.name, 0) + 1
    ranked = sorted(counts, key=lambda name: (-counts[name], name))[:4]
    return [
        {"signal": {"type": "item", "name": name}, "index": index}
        for index, name in enumerate(ranked, start=1)
        if name in data.stack_sizes
    ]


def to_string(blueprint: Blueprint, data: Data | None = None, *, label: str = "") -> str:
    return encode_string(to_json(blueprint, data, label=label))


def rotate(
    blueprint: Blueprint,
    quarters: int,
    data: Data | None = None,
    *,
    normalise: bool = True,
) -> Blueprint:
    """Rotate by `quarters` * 90 degrees clockwise about the origin.

    Rotation is the cheapest honest augmentation there is: a smelter column is
    still a smelter column sideways, and it quadruples the corpus without
    teaching the model a single thing that is false.

    `normalise=False` leaves the result where the arithmetic put it, negative
    coordinates and all. Anything travelling alongside the design — a module's
    ports, say — has to be shifted by the *same* offset, and that is only
    possible if the shift has not already happened.
    """
    data = data or load()
    quarters %= 4
    out = []
    for placement in blueprint.entities:
        width, height = placement.size(data)
        x, y = placement.x, placement.y
        for _ in range(quarters):
            x, y, width, height = -(y + height), x, height, width
        direction = (placement.direction + 2 * quarters) % 8
        out.append(replace(placement, x=x, y=y, direction=direction))
    turned = Blueprint(entities=out, label=blueprint.label, source=blueprint.source)
    return turned.normalised(data) if normalise else turned


# Mirroring maps a direction to its reflection. Belts and inserters keep working
# because north/south are fixed under a horizontal flip and east/west swap.
_FLIP_X = {NORTH: NORTH, EAST: WEST, SOUTH: SOUTH, WEST: EAST, 1: 7, 3: 5, 5: 3, 7: 1}


def mirror(blueprint: Blueprint, data: Data | None = None, *, normalise: bool = True) -> Blueprint:
    """Reflect left-to-right.

    Splitters and underground belts survive this; anything asymmetric in a way
    the tile grid cannot see (rail signals, say) does not, which is why the
    augmentation pipeline skips blueprints containing rails.
    """
    data = data or load()
    out = []
    for placement in blueprint.entities:
        direction = _FLIP_X.get(placement.direction, placement.direction)
        width, _ = placement.size(data)
        out.append(replace(placement, x=-(placement.x + width), direction=direction))
    flipped = Blueprint(entities=out, label=blueprint.label, source=blueprint.source)
    return flipped.normalised(data) if normalise else flipped
