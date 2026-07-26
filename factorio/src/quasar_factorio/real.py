"""Human blueprints, from a fetched cache into the corpus.

The generators saturate. 35,200 draws through `synth.sample` produce 7,579
distinct layouts, and six of the eleven generators are exhausted by a few
hundred — a lab block has 56 forms and a mining outpost 44. That ceiling is
about 12M tokens, and the Chinchilla budget for `factorio-nano` is 70.4M: six passes
over everything the generators can say, when data-constrained scaling laws put
the point where repeating stops being nearly free at four. Blueprints people
published are the only source of layouts nobody wrote a generator for.

`tools/fetch_blueprints.py` writes the cache; this module is the half that
decides what a blueprint has to be to join the corpus. Four things get thrown
out, in the order that makes the counts readable:

*Strings this harness cannot read.* 185 of the archive's 17,477 records are the
pre-0.15 gzip format, which `decode_string` refuses on purpose — all of them at
the oldest end, so a crawl that took only the earliest ids would meet far more
of them than 1%.

*Designs that are not vanilla 1.1.* The prototype table is 1.1's, so a Space
Age or modded blueprint arrives as a few recognisable belts around entities
that do not exist. Importing that would be importing a hole. A blueprint is
kept only if `MIN_KNOWN` of its entities are in the table — not "the known ones
are kept", which is how you end up training on rubble.

*Designs that do not fit.* 64x64 and 64 entities is the grid the grammar can
address. A base is not croppable into one: a cropped design is a truncated
design, and `augment` exists precisely because a corpus that contains broken
output teaches that broken output is sometimes correct.

*Curved rails.* Their footprint is not a rectangle, so the tile model this
harness uses genuinely cannot represent them, and every overlap check involving
one is wrong in both directions. Straight rails are fine and stay.

What survives is graded by exactly the same `validate.grade` call the synthetic
half goes through, augmented by the same symmetries, and split by the same
layout key — a real balancer and its mirror image cannot land on opposite sides
of the split any more than a generated one can.
"""

from __future__ import annotations

import collections
import json
import pathlib
import random
from collections.abc import Iterator
from dataclasses import replace

from . import augment, grammar, validate
from . import blueprint as bp
from .dataset import Design
from .grammar import COUNT_MAX, Spec
from .prototypes import Data, load

#: Anything smaller is a fragment — one chest, a lone pole, half a belt — and
#: contributes nothing but noise to a corpus of designs.
MIN_ENTITIES = 4

#: Fraction of a blueprint's entities that must exist in the 1.1 table for the
#: rest to be worth keeping. At 0.9 a 40-entity design may carry four modded
#: lamps and still come in; at 0.5 it would come in as a ruin.
MIN_KNOWN = 0.9

#: Prototypes whose real footprint is not the rectangle the tile model assumes.
UNMODELLED = frozenset({"curved-rail"})

#: Recipes renamed by Factorio 0.17. The blueprints that use the old spellings
#: are otherwise perfectly good, and the alternative to a five-line table is
#: throwing away every science block uploaded before 2019.
RENAMED = {
    "science-pack-1": "automation-science-pack",
    "science-pack-2": "logistic-science-pack",
    "science-pack-3": "chemical-science-pack",
    "high-tech-science-pack": "utility-science-pack",
    "raw-wood": "wood",
}

#: What the corpus calls a design that came from a person. Already in
#: `grammar.KINDS`; the model is meant to be able to tell the difference.
KIND = "real"


def records(path: pathlib.Path) -> Iterator[dict]:
    """Cache lines, oldest id first.

    Sorted rather than in file order: the crawler is threaded, so file order is
    whatever the network decided that afternoon, and a corpus that changes when
    the fetch is re-run is a corpus whose runs cannot be compared.
    """
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            lines.append(json.loads(line))
        except ValueError:  # a crawl killed mid-flush; see `fetch_blueprints`
            continue
    lines.sort(key=lambda record: str(record.get("id", "")))
    yield from lines


def designs(
    path: pathlib.Path,
    *,
    data: Data | None = None,
    limit: int = 0,
    variants: int = 4,
    seed: int = 0,
    counts: collections.Counter | None = None,
) -> Iterator[Design]:
    """Every usable design in the cache, expanded like a generated one.

    `counts` is filled in with the reason for each rejection, because "the real
    half contributed 3,000 designs" is not a number anyone can act on and "1,900
    were Space Age" is.
    """
    data = data or load()
    counts = collections.Counter() if counts is None else counts
    kept = 0
    for index, record in enumerate(records(path)):
        counts["records"] += 1
        rng = random.Random(seed * 1_000_003 + index)
        for design in _designs_of(record, data, rng, variants, counts):
            yield design
            kept += 1
            if limit and kept >= limit:
                return


def _designs_of(record, data, rng, variants, counts) -> Iterator[Design]:
    try:
        payload = bp.decode_string(str(record.get("string", "")))
    except bp.BlueprintError:
        counts["undecodable"] += 1
        return
    for entry in bp.books(payload):
        counts["blueprints"] += 1
        design = _design_of(entry, data, rng, variants, counts, record)
        if design is not None:
            counts["kept"] += 1
            yield design


def _design_of(entry, data, rng, variants, counts, record) -> Design | None:
    """One `blueprint` object, or `None` with the reason counted."""
    raw = entry.get("entities") or []
    if not raw:
        counts["no entities"] += 1
        return None
    known = sum(1 for entity in raw if data.entity(entity.get("name")) is not None)
    if known < MIN_ENTITIES:
        counts["too small"] += 1
        return None
    if known / len(raw) < MIN_KNOWN:
        counts["modded or 2.0"] += 1
        return None
    if any(entity.get("name") in UNMODELLED for entity in raw):
        counts["unmodelled geometry"] += 1
        return None

    try:
        design = _renamed(bp.from_json(entry, data), data)
    except bp.BlueprintError:
        counts["unconvertible"] += 1
        return None
    if len(design.entities) > COUNT_MAX:
        counts["too many entities"] += 1
        return None
    if not design.fits(data):
        counts["off grid"] += 1
        return None

    spec = Spec.measure(design, KIND, product(design, data), data)
    documents = []
    for form, form_spec in augment.variants(design, spec, data, rng=rng, limit=variants):
        try:
            text = grammar.serialise(form, data, form_spec)
        except bp.BlueprintError:
            continue
        if validate.grade(text, data).valid:
            documents.append(text)
    if not documents:
        # Graded here as well as in `dataset.build` so that the *reason* is
        # counted against the source rather than showing up as an anonymous
        # rejection at the end of a corpus build.
        counts["invalid"] += 1
        return None
    design.label = str(record.get("title") or "")
    design.source = str(record.get("id") or "")
    return Design(kind=KIND, blueprint=design, spec=spec, documents=documents)


def _renamed(design: bp.Blueprint, data: Data) -> bp.Blueprint:
    """Old recipe spellings brought forward; unknown ones dropped, not kept.

    A recipe the 1.1 table has never heard of fails validation for the whole
    blueprint, and an assembler with no recipe at all is a thing the game builds
    happily — so dropping the field keeps the design and loses only the hint.
    """
    out = []
    for placement in design.entities:
        recipe = RENAMED.get(placement.recipe, placement.recipe)
        if recipe is not None and recipe not in data.recipes:
            recipe = None
        out.append(replace(placement, recipe=recipe))
    return bp.Blueprint(entities=out, label=design.label, source=design.source)


def product(design: bp.Blueprint, data: Data) -> str | None:
    """What the design is for, as far as its machines say.

    The most common recipe among its crafters. A blueprint of nothing but belts
    has no answer and gets `r:none`, which is the same thing the generators emit
    for a balancer.
    """
    counts: collections.Counter[str] = collections.Counter()
    for placement in design.entities:
        if placement.recipe:
            counts[placement.recipe] += 1
    if not counts:
        return None
    best = max(counts.values())
    # Ties by name, so the spec of a design does not depend on entity order.
    return min(name for name, seen in counts.items() if seen == best)
