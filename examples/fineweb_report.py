#!/usr/bin/env python3
"""Turn a long FineWeb-Edu training log into a small, strict evidence file."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path
from typing import NamedTuple


TRAIN = re.compile(
    r"^optimizer step (?P<step>\d+)/(?P<steps>\d+) \| "
    r"loss (?P<loss>\S+) .* \| (?P<throughput>\d+) tok/s \|"
)
VALID = re.compile(
    r"^\s+valid: step (?P<step>\d+) \| loss (?P<loss>\S+) .* "
    r"\| bpb (?P<bpb>\S+) \|"
)
FINAL = re.compile(
    r"^final: step (?P<step>\d+) \| loss (?P<loss>\S+) .* "
    r"\| bpb (?P<bpb>\S+) \|"
)


class TrainingPoint(NamedTuple):
    step: int
    loss: float
    throughput: int


class Report(NamedTuple):
    elapsed_seconds: int
    expected_steps: int
    final_step: int | None
    minimum_throughput: int
    median_throughput: int | None
    throughput_samples: int
    early_train_loss: float | None
    late_train_loss: float | None
    first_validation_bpb: float | None
    final_validation_bpb: float | None
    nonfinite_events: int
    errors: tuple[str, ...]


def number(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {text}")
    return value


def edge_median(points: list[TrainingPoint], *, first: bool) -> float | None:
    if not points:
        return None
    width = min(5, max(1, len(points) // 2))
    edge = points[:width] if first else points[-width:]
    return statistics.median(point.loss for point in edge)


def build_report(
    path: Path,
    *,
    elapsed_seconds: int,
    expected_steps: int,
    minimum_throughput: int,
) -> Report:
    training: list[TrainingPoint] = []
    validation_bpb: list[float] = []
    final_step = None
    final_validation_bpb = None
    nonfinite_events = 0
    errors: list[str] = []

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip("\r")
        if match := TRAIN.match(line):
            training.append(
                TrainingPoint(
                    step=int(match["step"]),
                    loss=number(match["loss"]),
                    throughput=int(match["throughput"]),
                )
            )
        elif match := VALID.match(line):
            validation_bpb.append(number(match["bpb"]))
        elif match := FINAL.match(line):
            final_step = int(match["step"])
            final_validation_bpb = number(match["bpb"])

        if line.startswith("non-finite gradient"):
            nonfinite_events += 1
        if line.startswith("Error:"):
            errors.append(line)

    # The first tenth contains compilation/autotuning and is not steady-state
    # throughput. Eval windows can be slow too, but the median makes those
    # occasional samples visible without letting them define the result.
    steady = [point for point in training if point.step >= expected_steps // 10]
    throughputs = [point.throughput for point in steady]
    median_throughput = round(statistics.median(throughputs)) if throughputs else None

    return Report(
        elapsed_seconds=elapsed_seconds,
        expected_steps=expected_steps,
        final_step=final_step,
        minimum_throughput=minimum_throughput,
        median_throughput=median_throughput,
        throughput_samples=len(throughputs),
        early_train_loss=edge_median(steady, first=True),
        late_train_loss=edge_median(steady, first=False),
        first_validation_bpb=validation_bpb[0] if validation_bpb else None,
        final_validation_bpb=final_validation_bpb,
        nonfinite_events=nonfinite_events,
        errors=tuple(errors),
    )


def validate(report: Report) -> None:
    if report.final_step != report.expected_steps:
        raise ValueError(
            f"training ended at step {report.final_step}, expected {report.expected_steps}"
        )
    if report.errors:
        raise ValueError(f"training printed an error: {report.errors[0]}")
    if report.nonfinite_events:
        raise ValueError(
            f"training reported {report.nonfinite_events} non-finite gradient event(s)"
        )
    if report.throughput_samples < 4 or report.median_throughput is None:
        raise ValueError("not enough steady-state throughput samples")
    if report.median_throughput < report.minimum_throughput:
        raise ValueError(
            f"median throughput {report.median_throughput} is below "
            f"{report.minimum_throughput} tok/s"
        )
    if report.early_train_loss is None or report.late_train_loss is None:
        raise ValueError("training loss curve is missing")
    if report.late_train_loss >= report.early_train_loss:
        raise ValueError(
            f"training loss did not fall: {report.early_train_loss:.4f} -> "
            f"{report.late_train_loss:.4f}"
        )
    if report.first_validation_bpb is None or report.final_validation_bpb is None:
        raise ValueError("validation bpb curve is missing")
    if report.final_validation_bpb >= report.first_validation_bpb:
        raise ValueError(
            f"validation bpb did not fall: {report.first_validation_bpb:.4f} -> "
            f"{report.final_validation_bpb:.4f}"
        )


def render(report: Report) -> str:
    def metric(value: float | int | None, precision: int = 4) -> str:
        if value is None:
            return "none"
        if isinstance(value, int):
            return str(value)
        return f"{value:.{precision}f}"

    return "\n".join(
        (
            "dataset=HuggingFaceFW/fineweb-edu/sample/10BT",
            f"elapsed_seconds={report.elapsed_seconds}",
            f"elapsed_hours={report.elapsed_seconds / 3600:.3f}",
            f"final_step={metric(report.final_step)}",
            f"expected_steps={report.expected_steps}",
            f"median_throughput_tok_s={metric(report.median_throughput)}",
            f"minimum_throughput_tok_s={report.minimum_throughput}",
            f"throughput_samples={report.throughput_samples}",
            f"early_train_loss={metric(report.early_train_loss)}",
            f"late_train_loss={metric(report.late_train_loss)}",
            f"first_validation_bpb={metric(report.first_validation_bpb)}",
            f"final_validation_bpb={metric(report.final_validation_bpb)}",
            f"nonfinite_events={report.nonfinite_events}",
            f"errors={len(report.errors)}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--minimum-throughput", type=int, default=17_000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = build_report(
        args.log,
        elapsed_seconds=args.elapsed_seconds,
        expected_steps=args.expected_steps,
        minimum_throughput=args.minimum_throughput,
    )
    text = render(report) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    validate(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
