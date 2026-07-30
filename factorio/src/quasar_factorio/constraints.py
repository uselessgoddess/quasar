"""The finite blueprint grammar used by constrained decoding.

The model's tokenizer says which strings are tokens. This schema adds the facts
the tokenizer cannot carry: an entity's footprint, whether another field follows
its direction, and which recipes that machine is allowed to run. Rust loads the
JSON once beside a checkpoint and masks impossible next tokens before sampling.
"""

from __future__ import annotations

import json
import pathlib

from .grammar import NO_RECIPE, recipe_token
from .prototypes import Data, load

VERSION = "factorio-v1"


def build(data: Data | None = None) -> dict:
    """Return the portable grammar and geometry schema."""
    data = data or load()
    entities = {}
    for name, entity in sorted(data.entities.items()):
        recipes = []
        if entity.takes_recipe:
            recipes = [
                NO_RECIPE,
                *(
                    recipe_token(recipe.name)
                    for recipe in sorted(data.recipes.values(), key=lambda recipe: recipe.name)
                    if recipe.category in entity.crafting_categories
                ),
            ]
        entities[name] = {
            "width": entity.width,
            "height": entity.height,
            "recipes": recipes,
            "flow": entity.takes_flow,
        }
    return {"version": VERSION, "entities": entities}


def write(path: pathlib.Path, data: Data | None = None) -> None:
    """Write the schema next to the tokenizer it describes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build(data), indent=2, sort_keys=True) + "\n")
