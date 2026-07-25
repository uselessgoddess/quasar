"""Human blueprints from a cache: what comes in, and what is refused.

The rejections matter more than the acceptances here. A filter that lets a
Space Age blueprint through as "the six belts we recognised" quietly poisons a
corpus in a way no validity metric would catch — every document still parses.
"""

import collections
import json

import pytest

from quasar_factorio import blueprint as bp
from quasar_factorio import dataset, grammar, prototypes, real, validate
from quasar_factorio.blueprint import Blueprint, Placement
from quasar_factorio.prototypes import EAST, NORTH

DATA = prototypes.load()


def belts(count: int = 8, step: int = 1) -> Blueprint:
    return Blueprint(
        entities=[Placement("transport-belt", x * step, 0, EAST) for x in range(count)]
    ).normalised(DATA)


def cache(tmp_path, *payloads, name: str = "blueprints.jsonl"):
    """A cache file in the shape `tools/fetch_blueprints.py` writes."""
    path = tmp_path / name
    lines = []
    for index, payload in enumerate(payloads):
        string = payload if isinstance(payload, str) else bp.encode_string(payload)
        lines.append(
            json.dumps(
                {
                    "id": f"-K{index:04d}",
                    "source": "test",
                    "title": f"design {index}",
                    "author": "somebody",
                    "tags": [],
                    "favourites": 0,
                    "created": "0",
                    "string": string,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def harvest(path, **kwargs):
    counts = kwargs.pop("counts", None)
    return list(real.designs(path, data=DATA, counts=counts, **kwargs))


def test_a_cached_blueprint_becomes_a_design(tmp_path):
    path = cache(tmp_path, bp.to_json(belts(), DATA, label="a lane"))

    got = harvest(path)

    assert len(got) == 1
    design = got[0]
    assert design.kind == real.KIND == "real"
    assert design.spec.kind == "real"
    assert design.spec.count == 8  # no crafters, so the entity count
    assert design.documents
    assert design.blueprint.source == "-K0000"


def test_every_document_a_design_brings_is_one_the_game_would_build(tmp_path):
    path = cache(tmp_path, bp.to_json(belts(), DATA))

    for design in harvest(path):
        for text in design.documents:
            report = validate.grade(text, DATA)
            assert report.valid, text
            assert report.spec_kind == "real"


def test_a_book_contributes_every_blueprint_in_it(tmp_path):
    """A third of the site is books, and the blocks inside them are the point."""
    book = {
        "blueprint_book": {
            "blueprints": [
                bp.to_json(belts(6), DATA),
                bp.to_json(belts(5, step=2), DATA),
            ]
        }
    }
    path = cache(tmp_path, book)

    counts = collections.Counter()
    got = harvest(path, counts=counts)

    assert len(got) == 2
    assert counts["blueprints"] == 2


def test_a_mostly_modded_blueprint_is_refused_whole(tmp_path):
    """Not "keep the six belts we recognised" — that imports a hole."""
    payload = {
        "blueprint": {
            "entities": [
                *(
                    {"name": "transport-belt", "position": {"x": index + 0.5, "y": 0.5}}
                    for index in range(5)
                ),
                *(
                    {"name": "some-mod-machine", "position": {"x": index + 6.5, "y": 0.5}}
                    for index in range(3)
                ),
            ]
        }
    }
    path = cache(tmp_path, payload)

    counts = collections.Counter()
    assert harvest(path, counts=counts) == []
    assert counts["modded or 2.0"] == 1


def test_a_fragment_is_not_a_design(tmp_path):
    path = cache(tmp_path, bp.to_json(belts(2), DATA))

    counts = collections.Counter()
    assert harvest(path, counts=counts) == []
    assert counts["too small"] == 1


def test_a_design_bigger_than_the_grammar_can_address_is_left_out(tmp_path):
    """64 entities and 64 tiles are the grammar's limits, not preferences."""
    path = cache(
        tmp_path,
        bp.to_json(belts(grammar.COUNT_MAX + 4), DATA),
        bp.to_json(belts(20, step=4), DATA),
    )

    counts = collections.Counter()
    assert harvest(path, counts=counts) == []
    assert counts["too many entities"] == 1
    assert counts["off grid"] == 1


def test_curved_rails_are_left_out(tmp_path):
    """Their footprint is not a rectangle, so every overlap check is a guess."""
    payload = {
        "blueprint": {
            "entities": [
                {"name": "curved-rail", "position": {"x": 4.0, "y": 4.0}, "direction": 1},
                *(
                    {"name": "straight-rail", "position": {"x": 1.0, "y": index * 2 + 1.0}}
                    for index in range(6)
                ),
            ]
        }
    }
    path = cache(tmp_path, payload)

    counts = collections.Counter()
    assert harvest(path, counts=counts) == []
    assert counts["unmodelled geometry"] == 1


def test_old_recipe_spellings_are_brought_forward(tmp_path):
    """0.17 renamed the science packs; the labs built before it still stand."""
    design = Blueprint(
        entities=[
            Placement("assembling-machine-2", index * 3, 0, NORTH, recipe="science-pack-1")
            for index in range(4)
        ]
    ).normalised(DATA)
    payload = bp.to_json(design, DATA)
    path = cache(tmp_path, payload)

    got = harvest(path)

    assert len(got) == 1
    assert got[0].spec.product == "automation-science-pack"
    recipes = {placement.recipe for placement in got[0].blueprint.entities}
    assert recipes == {"automation-science-pack"}


def test_a_recipe_nobody_has_heard_of_costs_the_hint_not_the_design(tmp_path):
    design = Blueprint(
        entities=[
            Placement("assembling-machine-2", index * 3, 0, NORTH, recipe="mod-widget")
            for index in range(4)
        ]
    ).normalised(DATA)
    path = cache(tmp_path, bp.to_json(design, DATA))

    got = harvest(path)

    assert len(got) == 1
    assert got[0].spec.product is None
    assert all(placement.recipe is None for placement in got[0].blueprint.entities)


def test_an_unreadable_string_is_counted_rather_than_raised(tmp_path):
    """A third of the oldest uploads predate the format `decode_string` reads."""
    path = cache(tmp_path, "H4sIAAAAAAAA/6tWKkstKlayUrBSMDIwMjIwMDA0MDAyMA==")

    counts = collections.Counter()
    assert harvest(path, counts=counts) == []
    assert counts["undecodable"] == 1


def test_a_broken_last_line_does_not_lose_the_rest(tmp_path):
    """The crawler appends as it goes, so a killed run leaves half a line."""
    path = cache(tmp_path, bp.to_json(belts(), DATA))
    with path.open("a") as sink:
        sink.write('{"id": "-K9999", "stri')

    assert len(harvest(path)) == 1


def test_the_harvest_does_not_depend_on_the_order_the_crawler_wrote_in(tmp_path):
    """The crawl is threaded; the corpus it feeds still has to be reproducible."""
    payloads = [bp.to_json(belts(count), DATA) for count in (6, 7, 8)]
    forward = cache(tmp_path, *payloads, name="forward.jsonl")
    lines = forward.read_text().splitlines()
    (tmp_path / "shuffled.jsonl").write_text("\n".join(reversed(lines)) + "\n")

    first = [design.documents for design in harvest(forward)]
    second = [design.documents for design in harvest(tmp_path / "shuffled.jsonl")]

    assert first == second


def test_the_limit_stops_the_harvest_early(tmp_path):
    path = cache(tmp_path, *(bp.to_json(belts(count), DATA) for count in (5, 6, 7, 8)))

    assert len(harvest(path, limit=2)) == 2


@pytest.mark.parametrize("variants", [1, 4])
def test_human_designs_join_the_corpus_on_the_same_terms(tmp_path, variants):
    """The whole point of `dataset.build(extra=...)`: one mixture, one grader."""
    path = cache(tmp_path, *(bp.to_json(belts(count), DATA) for count in (6, 7, 8)))
    out = tmp_path / "corpus"

    stats = dataset.build(
        out,
        8,
        seed=1,
        variants=variants,
        data=DATA,
        extra=real.designs(path, data=DATA, variants=variants),
    )

    assert stats.kinds.get("real") == 3
    assert stats.rejected == 0
    assert sum(stats.kinds.values()) == stats.designs
