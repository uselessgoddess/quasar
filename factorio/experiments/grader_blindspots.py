"""How much can a design score 1.0 on every local metric and still not work?

This is the measurement the issue's training log demanded. That run finished
with parses 1.000, valid 0.979, powered 1.000, belts 1.000, inserters 0.995 and
quality 0.977 — six numbers at the ceiling and nothing left to optimise. The
question is whether the benchmark had been solved or had merely stopped looking,
and the way to answer it is to take designs the grader calls perfect and check
whether items can reach the machines.

The perturbations here are the smallest edits that keep every local metric
intact and break the factory:

*wrong recipe* — retarget one assembler to something the belts do not carry. It
is still a legal recipe for that machine, still powered, still surrounded by
connected inserters. It is fed nothing.

*turned inserter* — flip one inserter end for end. It still has an entity on
both sides, so `connected_inserters` cannot tell, but it now takes from the
machine and puts back onto the supply belt.

*severed output* — delete the inserter that takes the product away. Everything
left is connected, and the machine crafts one batch and stops.

Run:  PYTHONPATH=src python experiments/grader_blindspots.py [draws]

Prints, per perturbation, how many designs the old metrics still call perfect
and what the flow grader says about the same designs. A large gap in the
`QUALITY` column against the `FLOW` column is the blind spot, quantified.
"""

from __future__ import annotations

import copy
import random
import sys

from quasar_factorio import prototypes, synth, validate
from quasar_factorio.blueprint import Blueprint
from quasar_factorio.prototypes import EAST, NORTH, SOUTH, WEST

# Turning an inserter round: north for south, east for west.
OPPOSITE = {NORTH: SOUTH, SOUTH: NORTH, EAST: WEST, WEST: EAST}


def _machines(blueprint: Blueprint, data) -> list[int]:
    return [
        index
        for index, placement in enumerate(blueprint.entities)
        if data.recipes.get(placement.recipe or "") is not None
    ]


def _inserters(blueprint: Blueprint, data) -> list[int]:
    return [
        index
        for index, placement in enumerate(blueprint.entities)
        if (proto := data.entity(placement.name)) is not None and proto.category == "inserter"
    ]


def untouched(blueprint, spec, data, rng):
    return blueprint, spec


def wrong_recipe(blueprint, spec, data, rng):
    """Retarget one machine to a legal recipe it is not being fed."""
    indices = _machines(blueprint, data)
    if not indices:
        return None
    index = rng.choice(indices)
    placement = blueprint.entities[index]
    proto = data.entity(placement.name)
    options = [
        recipe.name
        for recipe in data.recipes.values()
        if recipe.category in proto.crafting_categories and recipe.name != placement.recipe
    ]
    if not options:
        return None
    edited = copy.deepcopy(blueprint)
    edited.entities[index] = type(placement)(
        placement.name, placement.x, placement.y, placement.direction, recipe=rng.choice(options)
    )
    return edited, spec


def turned_inserter(blueprint, spec, data, rng):
    """Flip one inserter, so it moves items back the way they came."""
    indices = _inserters(blueprint, data)
    if not indices:
        return None
    index = rng.choice(indices)
    placement = blueprint.entities[index]
    turned = OPPOSITE.get(placement.direction)
    if turned is None:
        return None
    edited = copy.deepcopy(blueprint)
    edited.entities[index] = type(placement)(placement.name, placement.x, placement.y, turned)
    return edited, spec


def severed_output(blueprint, spec, data, rng):
    """Delete one inserter, so a machine crafts a batch and jams."""
    indices = _inserters(blueprint, data)
    if not indices:
        return None
    edited = copy.deepcopy(blueprint)
    del edited.entities[rng.choice(indices)]
    return edited, spec


DAMAGE = (untouched, wrong_recipe, turned_inserter, severed_output)


def main() -> int:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    data = prototypes.load()

    print(f"{'PERTURBATION':<18}{'N':>6}{'VALID':>8}{'QUALITY':>9}{'FLOW':>8}{'PERFECT':>9}")
    print("-" * 58)
    rows = []
    for damage in DAMAGE:
        reports = []
        for seed in range(draws):
            rng = random.Random(seed)
            blueprint, spec = synth.module(rng, data)
            edited = damage(blueprint, spec, data, random.Random(seed + 10_000))
            if edited is None:
                continue
            reports.append(validate.inspect(edited[0].normalised(data), data, edited[1]))
        if not reports:
            continue
        valid = [report for report in reports if report.valid]
        # "Perfect" is the old benchmark's ceiling: everything it can measure,
        # at 1.0. The flow column is what those same designs actually do.
        perfect = [report for report in valid if report.quality() == 1.0]
        flow = sum(report.flows() for report in perfect) / len(perfect) if perfect else 0.0
        rows.append((damage.__name__, len(reports), len(valid), perfect, flow))
        print(
            f"{damage.__name__:<18}{len(reports):>6}{len(valid) / len(reports):>8.3f}"
            f"{sum(r.quality() for r in reports) / len(reports):>9.3f}"
            f"{sum(r.flows() for r in reports) / len(reports):>8.3f}"
            f"{len(perfect) / len(reports):>9.3f}"
        )

    print(
        "\nPERFECT is the fraction scoring 1.0 on every metric the run in issue #17\n"
        "reported. FLOW is what the item-flow grader says about the whole column.\n"
    )
    for name, _, _, perfect, flow in rows:
        if name == "untouched":
            continue
        print(f"{name}: {len(perfect)} designs the old grader calls flawless, mean flow {flow:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
