"""How many *distinct* designs does each generator actually have?

`dataset.build` deduplicates by `augment.canonical`, so asking for 20,000
designs does not mean getting 20,000. A generator whose parameter space is
small saturates: past some point every new draw is a design already in the
corpus, and the only thing that grows is the duplicate counter. Training on a
corpus that is 40% one repeated belt lane is how a model learns to emit belt
lanes.

Run:  PYTHONPATH=src python experiments/saturation.py [draws]

Prints, per generator, the distinct-design count against draw count. A curve
that is still climbing has headroom; one that has flattened is telling you the
generator needs more parameters, not more samples.
"""

from __future__ import annotations

import random
import sys

from quasar_factorio import augment, prototypes, synth

CHECKPOINTS = (25, 50, 100, 200, 400, 800, 1600, 3200)


def curve(kind: str, draws: int, data) -> tuple[list[int], int]:
    seen: set[str] = set()
    counts = []
    for index in range(draws):
        rng = random.Random(index)
        blueprint, _ = synth.GENERATORS[kind](rng, data)
        seen.add(augment.canonical(blueprint, data))
        if index + 1 in CHECKPOINTS:
            counts.append(len(seen))
    return counts, len(seen)


def main() -> int:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    data = prototypes.load()
    points = [point for point in CHECKPOINTS if point <= draws]

    header = "".join(f"{point:>7}" for point in points)
    print(f"{'GENERATOR':<18}{header}{'YIELD':>9}")
    print("-" * (18 + len(header) + 9))

    totals = [0] * len(points)
    for kind in sorted(synth.GENERATORS):
        counts, final = curve(kind, draws, data)
        totals = [total + count for total, count in zip(totals, counts, strict=False)]
        row = "".join(f"{count:>7}" for count in counts)
        print(f"{kind:<18}{row}{final / draws:>8.0%}")

    row = "".join(f"{total:>7}" for total in totals)
    print(f"\n{'ALL (sum)':<18}{row}")
    print(
        "\nYIELD is distinct designs per draw at the last checkpoint. Anything\n"
        "well under 100% is a generator that has run out of things to say."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
