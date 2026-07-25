"""Turning one blueprint into several the game would accept just as happily.

Every augmentation here is a symmetry of Factorio itself, not a perturbation: a
mirrored smelter column smelts, a rotated bus tap taps, an upgraded belt lane
moves the same items faster. That matters more than the extra volume. A model
trained on jittered or truncated blueprints learns that broken output is
sometimes correct; a model trained on symmetries learns that a design is a
design regardless of which way it happens to be facing.

`retier` is the only non-geometric one, and it is deliberately one-directional.
Upgrading a yellow underground belt to a red one lengthens its reach, so a
tunnel that connected still connects; downgrading would silently sever it. Same
argument for assembling machines, whose recipe categories only ever grow with
the tier. Anything whose footprint changes with the tier — a stone furnace to an
electric one, 2x2 to 3x3 — is not in the table at all.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import replace

from .blueprint import Blueprint, mirror, rotate
from .grammar import Spec
from .prototypes import Data, load

# Same footprint, same or wider capability, same or longer reach. Checked
# against the prototype table by `tests/test_augment.py` rather than trusted.
UPGRADES = {
    "transport-belt": "fast-transport-belt",
    "fast-transport-belt": "express-transport-belt",
    "underground-belt": "fast-underground-belt",
    "fast-underground-belt": "express-underground-belt",
    "splitter": "fast-splitter",
    "fast-splitter": "express-splitter",
    "inserter": "fast-inserter",
    "fast-inserter": "stack-inserter",
    "assembling-machine-1": "assembling-machine-2",
    "assembling-machine-2": "assembling-machine-3",
    "stone-furnace": "steel-furnace",
}


#: The transitive bottom of each upgrade chain, for canonicalisation only.
BASELINE = {}
for _name in list(UPGRADES.values()):
    _root = _name
    while True:
        _lower = next((low for low, high in UPGRADES.items() if high == _root), None)
        if _lower is None:
            break
        _root = _lower
    BASELINE[_name] = _root


def canonical(blueprint: Blueprint, data: Data | None = None) -> str:
    """A key that is the same for every form of the same design.

    Two designs that differ only by a rotation, a reflection or a belt tier are
    the same design, and a corpus that treats them as two puts one of them in
    `train` and the other in `valid` — which is how a validation loss ends up
    measuring memorisation. Downgrading before comparing is safe here precisely
    because the result is never written anywhere: it is a hash, not a blueprint.
    """
    from .grammar import serialise  # local: grammar imports blueprint, not the reverse

    data = data or load()
    base = Blueprint(
        entities=[
            replace(placement, name=BASELINE.get(placement.name, placement.name))
            for placement in blueprint.entities
        ]
    )
    return min(serialise(form, data) for form in dihedral(base, data))


def retier(blueprint: Blueprint, data: Data | None = None) -> Blueprint | None:
    """Every upgradeable prototype bumped one tier, or `None` if none are.

    Applied to the whole blueprint at once, which is what keeps underground
    pairs matched: both halves carry the same name, so both move together.
    """
    data = data or load()
    if not any(placement.name in UPGRADES for placement in blueprint.entities):
        return None
    return Blueprint(
        label=blueprint.label,
        source=blueprint.source,
        entities=[
            replace(placement, name=UPGRADES.get(placement.name, placement.name))
            for placement in blueprint.entities
        ],
    ).normalised(data)


def dihedral(blueprint: Blueprint, data: Data | None = None) -> Iterator[Blueprint]:
    """The eight rigid motions of the square, in a fixed order.

    Duplicates are not filtered here: a symmetric design genuinely maps to
    itself, and the corpus builder deduplicates on the serialised text anyway.
    """
    data = data or load()
    for reflected in (blueprint, mirror(blueprint, data)):
        for quarters in range(4):
            yield rotate(reflected, quarters, data)


def variants(
    blueprint: Blueprint,
    spec: Spec,
    data: Data | None = None,
    *,
    rng: random.Random | None = None,
    limit: int = 4,
) -> list[tuple[Blueprint, Spec]]:
    """Up to `limit` distinct forms of one design, spec re-measured for each.

    Re-measuring rather than copying the spec is the whole point: rotating a
    3x9 smelter column makes it 9x3, and a spec that still claimed 3x9 would
    teach the model that the dimensions in the prompt are decorative.
    """
    data = data or load()
    rng = rng or random.Random(0)
    pool = list(dihedral(blueprint, data))
    if (upgraded := retier(blueprint, data)) is not None:
        pool += list(dihedral(upgraded, data))

    seen: set[tuple] = set()
    out: list[tuple[Blueprint, Spec]] = []
    for candidate in _shuffled_keeping_first(pool, rng):
        key = tuple(candidate.entities)
        if key in seen or not candidate.fits(data):
            continue
        seen.add(key)
        out.append((candidate, Spec.measure(candidate, spec.kind, spec.product, data)))
        if len(out) >= limit:
            break
    return out


def _shuffled_keeping_first(pool: list[Blueprint], rng: random.Random) -> list[Blueprint]:
    """Shuffle, but keep the original orientation first.

    A corpus that always contains the design as drawn, plus a random sample of
    its symmetries, stays comparable across `limit` settings — turning
    augmentation down removes variants rather than replacing them.
    """
    rest = pool[1:]
    rng.shuffle(rest)
    return [pool[0], *rest]
