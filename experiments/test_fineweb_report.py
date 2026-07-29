#!/usr/bin/env python3
"""Regression tests for the FineWeb-Edu soak-run report gate."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fineweb_report", ROOT / "examples" / "fineweb_report.py"
)
assert SPEC is not None and SPEC.loader is not None
fineweb_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fineweb_report)


def log(
    *,
    throughputs: tuple[int, ...] = (17_100, 17_300, 17_200, 17_400),
    final_step: int = 1_800,
    nonfinite: bool = False,
) -> str:
    lines = [
        "training plan: 1800 optimizer steps | 0.24B tokens | 131072 tokens/step",
        (
            "training precision: fp16 output head + fp16 FFN projections + "
            "fp16 Mamba projections | fp32 master/norm/residual/SSD state/logits/loss "
            "| loss scale 1024"
        ),
    ]
    for step, loss, throughput in zip(
        (200, 400, 1_600, final_step),
        (8.2, 7.8, 6.9, 6.7),
        throughputs,
        strict=True,
    ):
        lines.append(
            f"optimizer step {step}/1800 | loss {loss:.4f} | lr 3.00e-3 | "
            f"{throughput} tok/s | 6.65 TFLOP/s | 0.20B tokens | ETA 0.0h"
        )
        if step == 400:
            lines.append(
                "  valid: step 400 | loss 7.9000 | ppl 2697.28 | "
                "bpb 2.1000 | 100000 tokens"
            )
    if nonfinite:
        lines.append(
            "non-finite gradient at step 1200 from a finite loss 7.0000; "
            "skipped the update (1 in a row)"
        )
    lines.append(
        f"final: step {final_step} | loss 6.8000 | ppl 897.85 | "
        "bpb 1.7000 | 100000 tokens"
    )
    return "\n".join(lines) + "\n"


class FinewebReportTests(unittest.TestCase):
    def report(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fineweb.log"
            path.write_text(text)
            return fineweb_report.build_report(
                path,
                elapsed_seconds=13_800,
                expected_steps=1_800,
                minimum_throughput=17_000,
            )

    def test_a_complete_fast_finite_learning_run_passes(self):
        report = self.report(log())

        self.assertEqual(report.final_step, 1_800)
        self.assertEqual(report.median_throughput, 17_250)
        self.assertLess(report.late_train_loss, report.early_train_loss)
        self.assertLess(report.final_validation_bpb, report.first_validation_bpb)
        fineweb_report.validate(report)

    def test_a_nonfinite_gradient_fails_even_if_the_run_recovers(self):
        report = self.report(log(nonfinite=True))

        with self.assertRaisesRegex(ValueError, "non-finite"):
            fineweb_report.validate(report)

    def test_a_run_below_the_requested_throughput_fails(self):
        report = self.report(log(throughputs=(16_800, 16_900, 16_700, 16_800)))

        with self.assertRaisesRegex(ValueError, "throughput"):
            fineweb_report.validate(report)

    def test_an_incomplete_run_fails(self):
        report = self.report(log(final_step=1_700))

        with self.assertRaisesRegex(ValueError, "ended at step"):
            fineweb_report.validate(report)


if __name__ == "__main__":
    unittest.main()
