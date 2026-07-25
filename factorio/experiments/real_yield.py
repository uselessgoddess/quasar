#!/usr/bin/env python3
"""How much of a cache of human blueprints survives the harness's grid.

Answers the question `real.py` is designed around: of the blueprints people
actually upload, what fraction is a 64x64, <=64-entity, vanilla-1.1 design the
corpus can take as-is? Every rejection is counted separately, so the answer says
*which* limit is doing the throwing away rather than only how much is left.

The filter chain is `real.designs` itself rather than a copy of it. A copy is
how a measurement quietly stops describing the thing it is measuring.

    python factorio/experiments/real_yield.py factorio/data/blueprints.jsonl

On the 4,936-record cache this was written against: 20,792 blueprints walked,
3,079 designs kept in 1,581 distinct layouts, 8,237 documents, 1.48M tokens.
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quasar_factorio import augment, prototypes, real, tokenizer  # noqa: E402


def main() -> int:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "factorio/data/blueprints.jsonl")
    data = prototypes.load()
    encoder = tokenizer.Encoder(data)
    counts: collections.Counter[str] = collections.Counter()
    kinds: collections.Counter[str] = collections.Counter()
    layouts: set[str] = set()
    sizes: list[tuple[int, int, int]] = []
    documents = tokens = 0

    for design in real.designs(path, data=data, counts=counts):
        layouts.add(augment.canonical(design.blueprint, data))
        sizes.append((len(design.blueprint.entities), design.spec.width, design.spec.height))
        documents += len(design.documents)
        tokens += sum(len(encoder.encode(text)) for text in design.documents)
        for placement in design.blueprint.entities:
            kinds[data.entities[placement.name].category] += 1

    width = max(len(name) for name in counts) + 2
    for name, seen in counts.most_common():
        print(f"{name:<{width}}{seen:>8}")
    if not sizes:
        return 1

    print(f"\nkept {len(sizes)} designs in {len(layouts)} distinct layouts")
    print(f"  documents     {documents}")
    print(f"  tokens        {tokens}")
    print(f"  mean entities {sum(s[0] for s in sizes) / len(sizes):.1f}")
    print(
        f"  mean extent   {sum(s[1] for s in sizes) / len(sizes):.1f}"
        f" x {sum(s[2] for s in sizes) / len(sizes):.1f}"
    )
    print("\nentity mix:")
    total = sum(kinds.values())
    for category, seen in kinds.most_common(12):
        print(f"  {category:<22}{seen:>7}  {seen / total:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
