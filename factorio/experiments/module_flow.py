"""Does a generated module actually move items from its inputs to its output?

The legality grader in `validate` answers a local question — is every inserter
reachable, is every machine powered — and a design can pass all of it while the
last stage sits idle because nothing ever hands it its second ingredient. That
is not hypothetical: the stacked-band layout hands a stage's product to the row
immediately below it and nowhere else, and the first draw of three-stage modules
graded 1.000 on quality while scoring `fed=0.67`, one starved stage each. The
`plan.modules` linearity filter exists because of this script.

Run:  PYTHONPATH=src python experiments/module_flow.py [draws]

Prints one line per draw and a summary. `delivers` is the fraction of declared
output ports the traced item flow actually reaches; `fed` is the fraction of
stages that receive every ingredient; `leaks` counts items crossing the zone
edge outside a declared port. Anything below 1.00 in the first two columns is a
module the corpus should not contain.
"""

from __future__ import annotations

import random
import sys
from collections import Counter

from quasar_factorio import flow, grammar, prototypes, synth, validate


def main() -> int:
    draws = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    data = prototypes.load()

    print(f"{'#':>3} {'PRODUCT':<24}{'STAGES':>7}{'SIZE':>8}{'ENT':>5}{'TOK':>5}", end="")
    print(f"{'VALID':>7}{'DELIV':>7}{'FED':>6}{'WORK':>6}{'MIX':>5}{'LEAK':>6}{'ZONE':>6}")
    print("-" * 96)

    failures: Counter[str] = Counter()
    totals = {"delivers": 0.0, "fed": 0.0, "working": 0.0}
    for index in range(draws):
        rng = random.Random(index)
        blueprint, spec = synth.module(rng, data)
        text = grammar.serialise(blueprint, data, spec=spec)
        report = validate.inspect(blueprint, data, spec=spec)
        traced = flow.trace(blueprint, spec, data)

        for key in totals:
            totals[key] += getattr(traced, key)
        if not report.valid:
            failures["invalid"] += 1
        if traced.delivers < 1.0:
            failures["undelivered"] += 1
        if traced.fed < 1.0:
            failures["starved"] += 1
        if traced.leaks:
            failures["leaking"] += 1
        if not traced.within_zone:
            failures["outside zone"] += 1

        print(
            f"{index:>3} {spec.product:<24}{len(spec.plan):>7}"
            f"{f'{spec.width}x{spec.height}':>8}{len(blueprint.entities):>5}"
            f"{len(text.split()):>5}{str(report.valid):>7}"
            f"{traced.delivers:>7.2f}{traced.fed:>6.2f}{traced.working:>6.2f}"
            f"{traced.mixed:>5}{traced.leaks:>6}{str(traced.within_zone):>6}"
        )

    print("-" * 96)
    means = " ".join(f"{key}={value / draws:.3f}" for key, value in totals.items())
    print(f"mean over {draws} draws: {means}")
    print("failures: " + (", ".join(f"{k}={v}" for k, v in failures.items()) or "none"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
