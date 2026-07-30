"""Post-hoc rejection and an apples-to-apples inference comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import benchmark, validate
from .prototypes import Data, load


def select(samples: list[dict], data: Data | None = None) -> list[dict]:
    """Choose the first hard-valid attempt per prompt, or the final failure.

    Attempts are precomputed in one model process because checkpoint loading is
    much more expensive than generation. ``attempt_budget`` records that actual
    compute, while ``attempts_used`` records when online rejection could have
    stopped.
    """
    data = data or load()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for index, sample in enumerate(samples):
        key = str(sample.get("benchmark_prompt") or sample.get("prompt") or index)
        grouped[key].append(sample)
    if not grouped:
        raise ValueError("no rejection samples")

    selected = []
    budgets = {len(attempts) for attempts in grouped.values()}
    if len(budgets) != 1:
        raise ValueError(f"incomplete rejection groups: attempt counts {sorted(budgets)}")
    for key, attempts in grouped.items():
        attempts.sort(key=lambda sample: int(sample.get("replicate", 0)))
        seeds = [sample.get("sampling_seed") for sample in attempts]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"repeated sampling seed for {key}")
        reports = [validate.grade(sample["text"], data) for sample in attempts]
        accepted = next(
            (
                index
                for index, report in enumerate(reports)
                if report.valid and report.spec_honoured
            ),
            len(attempts) - 1,
        )
        record = dict(attempts[accepted])
        record.update(
            {
                "decoding": "reject-regenerate",
                "replicate": 0,
                "attempt_budget": len(attempts),
                "attempts_used": accepted + 1,
                "accepted": reports[accepted].valid and reports[accepted].spec_honoured,
            }
        )
        selected.append(record)
    return selected


def compare(
    constrained: list[dict],
    rejected: list[dict],
    data: Data | None = None,
    *,
    confidence: float = 0.95,
    iterations: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare one selected answer per fixed prompt, including actual compute."""
    data = data or load()

    def keyed(rows):
        out = {}
        for row in rows:
            key = row.get("benchmark_prompt")
            if not key or key in out:
                raise ValueError("comparison needs one unique row per benchmark prompt")
            out[key] = row
        return out

    left, right = keyed(constrained), keyed(rejected)
    if set(left) != set(right):
        raise ValueError("constrained and rejection prompt sets differ")

    methods = {}
    for name, rows in (("constrained", constrained), ("reject_regenerate", rejected)):
        reports = [validate.grade(row["text"], data) for row in rows]
        methods[name] = {
            "generated_attempts": sum(int(row.get("attempt_budget", 1)) for row in rows),
            "mean_attempts_used": sum(int(row.get("attempts_used", 1)) for row in rows) / len(rows),
            "accepted": sum(
                bool(row.get("accepted", report.valid and report.spec_honoured))
                for row, report in zip(rows, reports, strict=True)
            ),
            "summary": validate.summarise(reports).to_dict(),
            "benchmark": benchmark.evaluate(
                rows,
                data,
                confidence=confidence,
                iterations=iterations,
                seed=seed,
            ),
        }
    return {
        "prompts": len(left),
        "criterion": "first valid and spec-honouring attempt",
        "methods": methods,
    }
