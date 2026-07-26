"""Where the wall clock of a corpus build goes, and what the cheap wins are worth.

The corpus stage of `examples/factorio.sh` is pure Python, runs before the GPU
gets to do anything, and shares a 75-minute CI budget with a half-hour training
run. Adding the module generator made it slower — modules are the biggest
designs in the mixture and every design is rotated, mirrored and re-serialised
twenty-four times to be deduplicated — so it is worth knowing, in seconds, what
the pipeline actually spends its time on.

The answer is not the generators. It is `augment.canonical`: eight rigid motions
per design, each of which rebuilds every `Placement` three times over and
recomputes the bounding box twice. That is a few million small allocations per
build, and the two things that made them cheaper are the ones this measures:

* `Placement.moved` / `.renamed` in place of `dataclasses.replace`, which
  re-derives the field list and dispatches through keywords on every call;
* `Blueprint.bounds` as a running minimum rather than `min`/`max` over two
  collected lists;
* `Data.producers_of` cached, because `plan.solve` walks a crafting tree and
  scans the whole recipe table at every node of it.

Run:  PYTHONPATH=src python experiments/corpus_cost.py [designs]
"""

from __future__ import annotations

import pathlib
import random
import sys
import tempfile
import time
from dataclasses import replace

from quasar_factorio import augment, dataset, prototypes, synth
from quasar_factorio.blueprint import Blueprint, Placement

DRAWS = 400


def _replaced(blueprint: Blueprint) -> Blueprint:
    """`normalised`, written the way it was written before — for the comparison."""
    return Blueprint(entities=[replace(e, x=e.x + 1, y=e.y + 1) for e in blueprint.entities])


def _moved(blueprint: Blueprint) -> Blueprint:
    return Blueprint(entities=[e.moved(e.x + 1, e.y + 1) for e in blueprint.entities])


def _bounds_by_lists(blueprint: Blueprint, data: prototypes.Data):
    """`bounds`, written the way it was written before."""
    if not blueprint.entities:
        return 0, 0, 0, 0
    xs: list[int] = []
    ys: list[int] = []
    for placement in blueprint.entities:
        width, height = placement.size(data)
        xs += [placement.x, placement.x + width]
        ys += [placement.y, placement.y + height]
    return min(xs), min(ys), max(xs), max(ys)


def _timed(label: str, call, repeats: int) -> float:
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    spent = time.perf_counter() - started
    print(f"{label:<34}{spent * 1e6 / repeats:8.1f} us")
    return spent


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else DRAWS
    data = prototypes.load()

    print("== ONE PLACEMENT =======================================================")
    placement = Placement("transport-belt", 3, 4, prototypes.EAST)
    _timed("dataclasses.replace", lambda: replace(placement, x=1, y=2), 200_000)
    _timed("Placement.moved", lambda: placement.moved(1, 2), 200_000)

    blueprint, _ = synth.sample(random.Random(3), data)
    print()
    print(f"== ONE DESIGN ({len(blueprint.entities)} entities) ===============================")
    _timed("rebuild, via replace", lambda: _replaced(blueprint), 20_000)
    _timed("rebuild, via moved", lambda: _moved(blueprint), 20_000)
    _timed("bounds, via two lists", lambda: _bounds_by_lists(blueprint, data), 20_000)
    _timed("bounds, running minimum", lambda: blueprint.bounds(data), 20_000)
    _timed("canonical (24 rebuilds of it)", lambda: augment.canonical(blueprint, data), 200)

    print()
    print("== RECIPE LOOKUPS ======================================================")
    # The cache is on the shared `Data`, so the cold number has to be read off a
    # copy nothing else has touched.
    cold = prototypes.load(prototypes.TABLE)
    _timed("producers_of, warm", lambda: data.producers_of("electronic-circuit"), 20_000)
    print(f"{'producers_of, cold':<34}{_uncached(cold) * 1e6:8.1f} us")

    print()
    print(f"== A BUILD OF {count} DESIGNS ==========================================")
    out = pathlib.Path(tempfile.mkdtemp())
    started = time.perf_counter()
    stats = dataset.build(out, count, data=data)
    spent = time.perf_counter() - started
    print(f"{'wall clock':<34}{spent:8.1f} s")
    print(f"{'distinct layouts':<34}{stats.designs:8d}")
    print(f"{'documents':<34}{stats.documents:8d}")
    print(f"{'per design':<34}{spent * 1000 / max(1, stats.designs):8.1f} ms")
    print(f"{'extrapolated to 20,000 draws':<34}{spent * 20_000 / count:8.1f} s")
    return 0


def _uncached(data: prototypes.Data) -> float:
    """One `producers_of` on a table whose cache is empty, which is a one-shot."""
    started = time.perf_counter()
    data.producers_of("electronic-circuit")
    return time.perf_counter() - started


if __name__ == "__main__":
    raise SystemExit(main())
