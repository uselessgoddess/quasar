"""What does the module generator add to the corpus, in the units that matter?

The training budget question from issue #17 is not "is the model too small" but
"is there anything left to learn". `factorio-nano` is 3.5M parameters, so the
Chinchilla rule asks for 70.4M tokens, and data-constrained scaling puts the
point where repeating stops being nearly free at about four epochs — so a run at
budget needs ~17.6M *unique* tokens. The reported run had 9.17M and did 7.7
epochs. Enlarging the model without moving that number moves the run further into
the repetition regime, not further along the loss curve.

So this measures the module generator the way `saturation.py` measures the older
ten: distinct layouts per draw, tokens per document, and the unique-token total
the whole catalogue can produce. It also prints the per-product breakdown,
because the catalogue is not uniform — a two-stage transport belt is a small
design and a three-stage underground belt is nearly four hundred tokens.

Run:  PYTHONPATH=src python experiments/module_yield.py [draws]
"""

from __future__ import annotations

import random
import sys

from quasar_factorio import augment, grammar, plan, prototypes, synth

CHECKPOINTS = (25, 50, 100, 200, 400, 800, 1600, 3200)


def main() -> int:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    data = prototypes.load()
    catalogue = plan.modules(data)

    print(f"catalogue: {len(catalogue)} modules")
    depths: dict[int, int] = {}
    for module in catalogue:
        depths[module.depth] = depths.get(module.depth, 0) + 1
    for depth in sorted(depths):
        print(f"  {depths[depth]:>3} at depth {depth}")
    print()

    seen: set[str] = set()
    curve = []
    per_product: dict[str, list] = {}
    for index in range(draws):
        blueprint, spec = synth.module(random.Random(index), data)
        text = grammar.serialise(blueprint, data, spec)
        key = augment.canonical(blueprint, data)
        fresh = key not in seen
        seen.add(key)
        per_product.setdefault(spec.product, []).append((len(text.split()), fresh))
        if index + 1 in CHECKPOINTS:
            curve.append((index + 1, len(seen)))

    print(f"{'PRODUCT':<28}{'DRAWS':>7}{'DISTINCT':>10}{'MEAN TOK':>10}{'MAX TOK':>9}")
    print("-" * 64)
    for product in sorted(per_product, key=lambda name: -len(per_product[name])):
        rows = per_product[product]
        tokens = [length for length, _ in rows]
        print(
            f"{product:<28}{len(rows):>7}{sum(fresh for _, fresh in rows):>10}"
            f"{sum(tokens) / len(tokens):>10.0f}{max(tokens):>9}"
        )

    print(f"\n{'DRAWS':>8}{'DISTINCT':>10}{'YIELD':>8}")
    for at, count in curve:
        print(f"{at:>8}{count:>10}{count / at:>8.0%}")

    tokens = sum(length for rows in per_product.values() for length, fresh in rows if fresh)
    print(
        f"\n{len(seen)} distinct layouts in {draws} draws, {tokens:,} tokens before augmentation."
        "\nEach is then expanded into up to eight symmetries, so the corpus contribution is"
        "\nseveral times this — and unlike a belt lane, every one of them is a design whose"
        "\ncorrectness the flow grader can check."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
