"""The distilled table against the wiki.

These numbers are checked by hand at https://wiki.factorio.com. If a regenerated
`prototypes.json` breaks one of them, the dump changed and every geometry
decision downstream needs looking at again.
"""

from quasar_factorio import prototypes
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST


def test_provenance_records_where_the_table_came_from():
    data = prototypes.load()
    assert data.provenance["factorio_version"].startswith("1.1")
    assert data.provenance["source"].startswith("https://")
    assert len(data.provenance["sha256"]) == 64


def test_sizes_match_the_wiki():
    entities = prototypes.load().entities
    for name, size in {
        "assembling-machine-1": (3, 3),
        "assembling-machine-2": (3, 3),
        "assembling-machine-3": (3, 3),
        "stone-furnace": (2, 2),
        "steel-furnace": (2, 2),
        "electric-furnace": (3, 3),
        "electric-mining-drill": (3, 3),
        "lab": (3, 3),
        "transport-belt": (1, 1),
        "splitter": (2, 1),
        "inserter": (1, 1),
        "medium-electric-pole": (1, 1),
        "big-electric-pole": (2, 2),
        "substation": (2, 2),
        "oil-refinery": (5, 5),
        "chemical-plant": (3, 3),
        "solar-panel": (3, 3),
        "accumulator": (2, 2),
        "rocket-silo": (9, 9),
    }.items():
        assert (entities[name].width, entities[name].height) == size, name


def test_speeds_and_ranges_match_the_wiki():
    entities = prototypes.load().entities
    assert entities["transport-belt"].items_per_second() == 15.0
    assert entities["fast-transport-belt"].items_per_second() == 30.0
    assert entities["express-transport-belt"].items_per_second() == 45.0
    assert entities["assembling-machine-1"].crafting_speed == 0.5
    assert entities["assembling-machine-2"].crafting_speed == 0.75
    assert entities["assembling-machine-3"].crafting_speed == 1.25
    assert entities["electric-mining-drill"].mining_speed == 0.5
    assert entities["small-electric-pole"].supply_area == 2.5
    assert entities["medium-electric-pole"].supply_area == 3.5
    assert entities["medium-electric-pole"].wire_distance == 9.0
    assert entities["substation"].supply_area == 9.0
    assert entities["underground-belt"].max_distance == 5.0
    assert entities["express-underground-belt"].max_distance == 9.0
    assert entities["long-handed-inserter"].max_distance == 2.0


def test_splitters_turn_sideways_and_squares_do_not():
    entities = prototypes.load().entities
    splitter = entities["splitter"]
    assert splitter.footprint(NORTH) == (2, 1)
    assert splitter.footprint(SOUTH) == (2, 1)
    assert splitter.footprint(EAST) == (1, 2)
    assert splitter.footprint(WEST) == (1, 2)
    assert entities["assembling-machine-1"].footprint(EAST) == (3, 3)


def test_only_underground_belts_carry_a_flow_flag():
    entities = prototypes.load().entities
    flowing = {name for name, proto in entities.items() if proto.takes_flow}
    assert flowing == {
        "underground-belt",
        "fast-underground-belt",
        "express-underground-belt",
    }


def test_recipes_are_real_and_complete():
    data = prototypes.load()
    gear = data.recipes["iron-gear-wheel"]
    assert gear.ingredients == (("iron-plate", 2),)
    assert gear.results == (("iron-gear-wheel", 1),)
    assert gear.time == 0.5

    circuit = data.recipes["electronic-circuit"]
    assert dict(circuit.ingredients) == {"iron-plate": 1, "copper-cable": 3}

    # Fluid recipes are the ones a tier 1 assembler cannot run.
    assert data.recipes["sulfuric-acid"].category == "chemistry"
    assert "chemistry" not in data.entities["assembling-machine-1"].crafting_categories
    assert "chemistry" in data.entities["chemical-plant"].crafting_categories


def test_lookups_find_the_right_machines():
    data = prototypes.load()
    able = {entity.name for entity in data.crafters_for(data.recipes["iron-gear-wheel"])}
    assert {"assembling-machine-1", "assembling-machine-2", "assembling-machine-3"} <= able
    assert "chemical-plant" not in able

    plate = data.producers_of("iron-plate")
    assert plate and plate[0].category == "smelting"


def test_everything_a_recipe_names_is_either_stackable_or_a_fluid():
    """The item table is swept out of every prototype category, not out of `item`.

    Factorio files a gear under `item`, a science pack under `tool`, a piercing
    round under `ammo` and a car under `item-with-entity-data`. Distilling only
    `data.raw.item` left 61 of the 214 things vanilla recipes name with no stack
    size at all, and the harness reads a missing stack size as "needs a pipe" —
    so the omission did not surface as missing data, it surfaced as every
    science pack being a fluid. This is the assertion that would have caught it.
    """
    data = prototypes.load()
    for recipe in data.recipes.values():
        for name, _ in (*recipe.ingredients, *recipe.results):
            assert name in data.stack_sizes or name in data.fluids, name


def test_the_fluids_are_the_ones_the_dump_calls_fluids():
    data = prototypes.load()
    assert {"water", "steam", "crude-oil", "petroleum-gas", "lubricant"} <= data.fluids
    assert not data.fluids & set(data.stack_sizes)
    for pack in ("automation-science-pack", "logistic-science-pack", "military-science-pack"):
        assert data.stack_sizes[pack] == 200, pack


def test_the_table_is_a_shared_immutable_singleton():
    assert prototypes.load() is prototypes.load()
