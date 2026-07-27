"""Audit the recipe DAGs that the spatial module generator does not yet cover.

The arithmetic planner can solve more recipes than the current stack/fork/factory
layouts can place.  This experiment makes that gap explicit and also measures
whether each admitted factory target is a real family of canonical geometries.

Run:

    PYTHONPATH=src python3 experiments/dag_catalogue.py [draws] [candidates]
    PYTHONPATH=src python3 experiments/dag_catalogue.py --sheet ../docs/screenshots/dag-v3-forms.png
"""

from __future__ import annotations

import argparse
import pathlib
import random
from collections import Counter

from quasar_factorio import augment, benchmark, grammar, plan, prototypes, render, synth


def _features(unit: plan.Plan) -> tuple[int, int, int]:
    """Shared intermediates, maximum recipe arity, and maximum dependency depth."""
    made = {stage.product: stage for stage in unit.stages}
    consumers = Counter(
        ingredient
        for stage in unit.stages
        for ingredient in set(stage.ingredients)
        if ingredient in made
    )

    def depth(item: str) -> int:
        stage = made.get(item)
        if stage is None:
            return 0
        return 1 + max((depth(ingredient) for ingredient in stage.ingredients), default=0)

    return (
        sum(count > 1 for count in consumers.values()),
        max(len(set(stage.ingredients)) for stage in unit.stages),
        depth(unit.product),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("draws", nargs="?", type=int, default=64)
    parser.add_argument("candidates", nargs="?", type=int, default=20)
    parser.add_argument("--sheet", type=pathlib.Path)
    args = parser.parse_args()
    draws = args.draws
    limit = args.candidates
    data = prototypes.load()
    admitted = {(module.product, module.supply): module for module in plan.modules(data)}

    print("ADMITTED FACTORY LAYOUT YIELD")
    for target in admitted.values():
        if target.shape != "factory":
            continue
        samples = [synth.module_for(random.Random(seed), data, target) for seed in range(draws)]
        layouts = {augment.canonical(blueprint, data) for blueprint, _ in samples}
        entities = [len(blueprint.entities) for blueprint, _ in samples]
        tokens = [
            len(grammar.serialise(blueprint, data, spec).split()) + 1 for blueprint, spec in samples
        ]
        print(
            f"{target.product:<30} {len(layouts):>3}/{draws:<3} canonical layouts "
            f"({len(layouts) / draws:.0%} yield); "
            f"entities {min(entities)}-{max(entities)}, tokens {min(tokens)}-{max(tokens)}"
        )

    print("\nUNPLACED DAG CANDIDATES")
    print(f"{'PRODUCT':<30} {'STG':>3} {'SHR':>3} {'ARY':>3} {'DEP':>3}  SUPPLY")
    rows = []
    seen = set()
    for depth_limit in range(2, plan.MAX_DEPTH + 1):
        for product, supply in plan.chains(data, depth=depth_limit).items():
            try:
                unit = plan.solve(data, product, supply, rate=1e-9, depth=depth_limit)
            except plan.PlanError:
                continue
            used = tuple(
                item
                for item in unit.supply
                if any(item in stage.ingredients for stage in unit.stages)
            )
            identity = (product, used)
            if identity in seen or identity in admitted or len(unit.stages) < 2:
                continue
            seen.add(identity)
            shared, arity, depth = _features(unit)
            if shared or arity > 2:
                rows.append((shared, arity, len(unit.stages), product, depth, used))

    ranked = sorted(rows, key=lambda row: (-row[0], -row[1], row[2], row[3]))
    for shared, arity, stages, product, depth, supply in ranked[:limit]:
        print(f"{product:<30} {stages:>3} {shared:>3} {arity:>3} {depth:>3}  {','.join(supply)}")
    print(f"\nshowing {min(limit, len(ranked))} of {len(ranked)} candidates")
    if args.sheet is not None:
        cards = []
        for target in admitted.values():
            if target.shape != "factory":
                continue
            for index, form in enumerate(benchmark.DAG_FORMS):
                blueprint, _ = synth.module_for(
                    random.Random(index),
                    data,
                    target,
                    factory_form=form,
                )
                cards.append(
                    render.card(
                        blueprint,
                        data,
                        title=target.product,
                        lines=(form.name_for(target.product),),
                        accent=render.GOOD,
                    )
                )
        args.sheet.parent.mkdir(parents=True, exist_ok=True)
        args.sheet.write_bytes(render.sheet(cards, columns=len(benchmark.DAG_FORMS)).png())
        print(f"wrote {len(cards)} held-out forms to {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
