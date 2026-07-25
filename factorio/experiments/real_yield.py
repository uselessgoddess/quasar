#!/usr/bin/env python3
"""How much of a cache of human blueprints survives the harness's grid.

Answers the question `real.py` is designed around: of the blueprints people
actually upload, what fraction is a 64x64, <=64-entity, vanilla-1.1 design the
corpus can take as-is? Every rejection is counted separately, so the answer says
*which* limit is doing the throwing away rather than only how much is left.

    python factorio/experiments/real_yield.py factorio/data/blueprints.jsonl
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from quasar_factorio import blueprint as bp  # noqa: E402
from quasar_factorio import grammar, prototypes, validate  # noqa: E402

MIN_ENTITIES = 4
MIN_KNOWN = 0.9


def main() -> int:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "factorio/data/blueprints.jsonl")
    data = prototypes.load()
    counts: collections.Counter[str] = collections.Counter()
    sizes: list[tuple[int, int, int]] = []
    kinds: collections.Counter[str] = collections.Counter()

    for line in path.read_text().splitlines():
        record = json.loads(line)
        counts["records"] += 1
        try:
            payload = bp.decode_string(record["string"])
        except bp.BlueprintError:
            counts["undecodable"] += 1
            continue
        counts["decoded"] += 1
        entries = list(bp.books(payload))
        counts["blueprints"] += len(entries)
        for entry in entries:
            raw = entry.get("entities") or []
            if not raw:
                counts["empty"] += 1
                continue
            known = sum(1 for e in raw if data.entity(e.get("name")) is not None)
            if known < MIN_ENTITIES:
                counts["too small"] += 1
                continue
            if known / len(raw) < MIN_KNOWN:
                counts["modded or 2.0"] += 1
                continue
            try:
                design = bp.from_json(entry, data)
            except bp.BlueprintError:
                counts["unconvertible"] += 1
                continue
            if len(design.entities) > grammar.COUNT_MAX:
                counts["too many entities"] += 1
                continue
            if not design.fits(data):
                counts["off grid"] += 1
                continue
            spec = grammar.Spec.measure(design, "real", None, data)
            try:
                text = grammar.serialise(design, data, spec)
            except bp.BlueprintError:
                counts["unserialisable"] += 1
                continue
            report = validate.grade(text, data)
            if not report.valid:
                counts["invalid"] += 1
                counts[f"  why: {report.error or 'overlap'}"] += 1
                continue
            counts["kept"] += 1
            sizes.append((report.entities, report.width, report.height))
            for placement in design.entities:
                kinds[data.entities[placement.name].category] += 1

    width = max(len(name) for name in counts) + 2
    for name, seen in counts.items():
        print(f"{name:<{width}}{seen:>8}")
    if sizes:
        print(f"\nkept {len(sizes)} designs")
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
