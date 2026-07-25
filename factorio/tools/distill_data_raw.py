#!/usr/bin/env python3
"""Distil Factorio's `data.raw` into the small table this harness actually uses.

The upstream dump is ~6.7 MB of sprite paths, sound definitions and animation
frames; roughly 0.3% of it is load-bearing for blueprint geometry. This script
keeps that 0.3% and writes `factorio/assets/prototypes.json`, which *is*
committed, so the harness runs offline and every number in it is traceable to a
real prototype rather than to somebody's memory of the game.

Usage:

    python factorio/tools/distill_data_raw.py                # downloads data.raw
    python factorio/tools/distill_data_raw.py --raw /tmp/data-raw.json

Provenance is recorded in the output: source URL, Factorio version and the
sha256 of the exact bytes the table was derived from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import urllib.request

# The community mirror of the vanilla prototype dump. It is the same JSON the
# data.raw visualiser (https://jackhugh.github.io/factorio-data-raw-visualiser/)
# renders, so anything asserted here can be checked by eye in a browser.
SOURCE = "https://jackhugh.github.io/factorio-data-raw-visualiser/data.json"

# Prototype categories that can appear inside a blueprint and that this harness
# models. Anything outside this list is dropped from a blueprint during import
# rather than silently mapped onto something else.
CATEGORIES = [
    "assembling-machine",
    "furnace",
    "transport-belt",
    "underground-belt",
    "splitter",
    "inserter",
    "electric-pole",
    "pipe",
    "pipe-to-ground",
    "container",
    "logistic-container",
    "mining-drill",
    "lab",
    "beacon",
    "boiler",
    "generator",
    "storage-tank",
    "pump",
    "radar",
    "roboport",
    "solar-panel",
    "accumulator",
    "offshore-pump",
    "rocket-silo",
    "wall",
    "gate",
    "straight-rail",
    "curved-rail",
]

# Entities whose prototype exists but which cannot be built by a player (crash
# site debris, the tutorial logos, the ship wrecks). They pollute the `container`
# category in particular.
UNBUILDABLE = (
    "crash-site",
    "big-ship-wreck",
    "factorio-logo",
    "-equipment",
    "blue-chest",
    "red-chest",
)

# Palette for the textureless renderer. Chosen to survive being printed small
# and greyscale-ish: hue carries the *role*, not the tier. Tier is carried by
# lightness inside a role, so a belt bus reads as one thing at a glance.
PALETTE = {
    "assembling-machine": "#4aa3df",
    "furnace": "#d95f3b",
    "transport-belt": "#f2c14e",
    "underground-belt": "#c79a2f",
    "splitter": "#e8b04b",
    "inserter": "#8e6fd8",
    "electric-pole": "#9aa5b1",
    "pipe": "#3fa796",
    "pipe-to-ground": "#2f7f72",
    "container": "#a3703f",
    "logistic-container": "#c08457",
    "mining-drill": "#6b7f3a",
    "lab": "#b25fb2",
    "beacon": "#5fd0b2",
    "boiler": "#b04a4a",
    "generator": "#c96a2b",
    "storage-tank": "#3f7fa7",
    "pump": "#357f9a",
    "radar": "#7f7f9a",
    "roboport": "#5f9fd8",
    "solar-panel": "#2f4f7f",
    "accumulator": "#4f5f9f",
    "offshore-pump": "#2f8f8f",
    "rocket-silo": "#9a9a9a",
    "wall": "#8a8a8a",
    "gate": "#a0a0a0",
    "straight-rail": "#6a6a6a",
    "curved-rail": "#6a6a6a",
}

# Categories whose entities take a recipe in the blueprint JSON. Furnaces do not
# (they infer it from their input), which is exactly why a furnace column is
# cheaper to generate than an assembler column.
RECIPE_TAKERS = {"assembling-machine", "rocket-silo"}


def tiles(box: list[list[float]]) -> tuple[int, int]:
    """Tile footprint of a collision box.

    Factorio places an entity on `ceil(collision width) x ceil(collision height)`
    tiles, with a floor of one tile so that zero-collision entities (inserters,
    small poles) still occupy their square. Checked against known values:
    assembling-machine-2 (2.4 -> 3x3), transport-belt (0.8 -> 1x1), splitter
    (1.8 x 0.8 -> 2x1), big-electric-pole (1.3 -> 2x2).
    """
    (x0, y0), (x1, y1) = box
    return max(1, math.ceil(x1 - x0)), max(1, math.ceil(y1 - y0))


def watts(energy: str | None) -> float | None:
    """Parse Factorio's `"150kW"` energy strings into watts."""
    if not isinstance(energy, str):
        return None
    scale = {"k": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
    text = energy.removesuffix("W").removesuffix("J").strip()
    if text and text[-1] in scale:
        return float(text[:-1]) * scale[text[-1]]
    try:
        return float(text)
    except ValueError:
        return None


def ingredient(entry: object) -> tuple[str, float] | None:
    """Normalise both ingredient spellings into `(name, amount)`.

    1.1 recipes mix the array form `["iron-plate", 2]` with the table form
    `{"name": ..., "amount": ..., "type": "fluid"}` freely, sometimes inside the
    same recipe.
    """
    if isinstance(entry, list) and len(entry) == 2:
        return str(entry[0]), float(entry[1])
    if isinstance(entry, dict) and "name" in entry:
        return str(entry["name"]), float(entry.get("amount", 1))
    return None


def recipe_body(recipe: dict) -> dict | None:
    """The buildable half of a recipe prototype.

    1.1 splits recipes into `normal`/`expensive` difficulty variants; vanilla
    play is `normal`, so that is what the planner arithmetic uses.
    """
    body = recipe.get("normal") if isinstance(recipe.get("normal"), dict) else recipe

    results = []
    if "result" in body:
        results.append((str(body["result"]), float(body.get("result_count", 1))))
    for entry in body.get("results", []) or []:
        pair = ingredient(entry)
        if pair:
            results.append(pair)
    if not results:
        return None

    ingredients = [ingredient(entry) for entry in body.get("ingredients", []) or []]
    if any(pair is None for pair in ingredients):
        return None

    return {
        "category": recipe.get("category", "crafting"),
        "time": float(body.get("energy_required", 0.5)),
        "ingredients": [{"name": name, "amount": amount} for name, amount in ingredients],
        "results": [{"name": name, "amount": amount} for name, amount in results],
    }


def entities(raw: dict) -> dict:
    """Per-entity geometry and the few gameplay numbers the planner needs."""
    out = {}
    for category in CATEGORIES:
        for name, proto in sorted(raw.get(category, {}).items()):
            if any(mark in name for mark in UNBUILDABLE):
                continue
            box = proto.get("collision_box")
            if not box:
                continue
            width, height = tiles(box)
            entry = {
                "category": category,
                "width": width,
                "height": height,
                "color": PALETTE[category],
                "takes_recipe": category in RECIPE_TAKERS,
            }
            for field, key in (
                ("crafting_speed", "crafting_speed"),
                ("speed", "speed"),
                ("supply_area_distance", "supply_area"),
                ("maximum_wire_distance", "wire_distance"),
                ("max_distance", "max_distance"),
                ("mining_speed", "mining_speed"),
                ("energy_usage", "energy_usage"),
            ):
                value = proto.get(field)
                if field == "energy_usage":
                    value = watts(value)
                if value is not None:
                    entry[key] = value
            if pickup := proto.get("pickup_position"):
                # An inserter's reach is not a field: it is how far its pickup
                # tile sits from its own. One tile for most, two for the
                # long-handed one, and the connectivity metric depends on it.
                entry["max_distance"] = float(round(max(abs(value) for value in pickup)))
            if categories := proto.get("crafting_categories"):
                entry["crafting_categories"] = sorted(categories)
            if slots := proto.get("module_specification"):
                entry["module_slots"] = slots.get("module_slots", 0)
            if area := proto.get("resource_searching_radius"):
                entry["mining_area"] = area
            out[name] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=pathlib.Path, help="a local copy of data.raw JSON")
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "assets" / "prototypes.json",
    )
    args = parser.parse_args()

    if args.raw:
        blob = args.raw.read_bytes()
    else:
        print(f"downloading {SOURCE}", file=sys.stderr)
        with urllib.request.urlopen(SOURCE, timeout=120) as response:
            blob = response.read()

    raw = json.loads(blob)
    recipes = {}
    for name, proto in sorted(raw.get("recipe", {}).items()):
        body = recipe_body(proto)
        if body:
            recipes[name] = body

    table = {
        "provenance": {
            "source": SOURCE,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "factorio_version": raw.get("utility-constants", {})
            .get("default", {})
            .get("factorio_version", "1.1"),
            "note": "generated by factorio/tools/distill_data_raw.py; do not edit by hand",
        },
        "entities": entities(raw),
        "recipes": recipes,
        "items": {
            name: {"stack_size": proto.get("stack_size", 50)}
            for name, proto in sorted(raw.get("item", {}).items())
        },
    }

    args.out.write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    print(
        f"{args.out}: {len(table['entities'])} entities, {len(table['recipes'])} recipes",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
