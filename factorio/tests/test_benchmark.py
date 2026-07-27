"""The fixed evaluation set and its repeated-sampling statistics."""

import json
import random

import pytest

from quasar_factorio import augment, benchmark, plan, prototypes, synth
from quasar_factorio.cli import main

DATA = prototypes.load()


@pytest.fixture(scope="module")
def repeated_samples():
    selected = benchmark.cases(DATA)
    rows = []
    for replicate in range(2):
        for index, case in enumerate(selected):
            record = case.record()
            record |= {
                "text": case.reference,
                "replicate": replicate,
                "sampling_seed": 1_337 + replicate * len(selected) + index,
            }
            rows.append(record)
    return rows


@pytest.fixture(scope="module")
def repeated_dag_samples():
    selected = benchmark.dag_cases(DATA)
    rows = []
    for replicate in range(2):
        for index, case in enumerate(selected):
            record = case.record()
            record |= {
                "text": case.reference,
                "replicate": replicate,
                "sampling_seed": 7_331 + replicate * len(selected) + index,
            }
            rows.append(record)
    return rows


def test_every_target_is_present_and_forks_are_oversampled():
    selected = benchmark.cases(DATA)
    counts = {}
    shapes = {}
    for case in selected:
        counts[case.target_id] = counts.get(case.target_id, 0) + 1
        shapes[case.target_id] = case.shape

    assert len(selected) == 64
    assert len(counts) == 29
    assert min(counts.values()) >= 2
    assert all(counts[name] >= 3 for name, shape in shapes.items() if shape == "fork")


def test_the_benchmark_is_byte_stable_for_the_same_version_and_seed():
    first = [case.record() for case in benchmark.cases(DATA)]
    second = [case.record() for case in benchmark.cases(DATA)]
    assert first == second


def test_dag_benchmark_covers_every_held_out_form_for_both_factories():
    selected = benchmark.dag_cases(DATA)
    assert len(selected) == benchmark.DAG_SIZE
    assert {case.target_id for case in selected} == set(benchmark.DAG_TARGETS)
    assert {case.layout for case in selected} == {
        f"{product}:{form.name_for(product)}"
        for target in benchmark.DAG_TARGETS
        for product in (target.split("|", 1)[0],)
        for form in benchmark.DAG_FORMS
    }
    assert len({augment.canonical(case.blueprint, DATA) for case in selected}) == (
        len(benchmark.DAG_TARGETS) * len(benchmark.DAG_FORMS)
    )
    assert len({case.record()["prompt"] for case in selected}) == len(selected)
    assert all(case.benchmark == benchmark.DAG_VERSION for case in selected)
    assert benchmark.DAG_VARIANTS == 2


@pytest.mark.parametrize("name", benchmark.DAG_TARGETS)
def test_dag_holdout_leaves_twenty_four_route_forms_per_target_for_training(name):
    target = next(target for target in plan.modules(DATA) if benchmark.target_id(target) == name)
    reserved = benchmark.reserved(benchmark.dag_cases(DATA), DATA)
    available = {
        augment.canonical(
            synth.module_for(random.Random(seed), DATA, target, factory_form=form)[0],
            DATA,
        )
        for seed, form in enumerate(synth.FACTORY_FORMS)
        if form not in benchmark.DAG_FORMS
    }
    assert len(available) == 24
    assert not reserved & available


def test_a_perfect_dag_run_reports_every_layout(repeated_dag_samples):
    result = benchmark.evaluate(repeated_dag_samples, DATA, iterations=20)
    assert (result["samples"], result["prompts"], result["factory_prompts"]) == (64, 32, 32)
    assert len(result["by_target"]) == 2
    assert len(result["by_layout"]) == 16
    assert result["summary"]["mean_flow"] == 1.0


def test_a_perfect_repeated_run_has_a_tight_perfect_interval(repeated_samples):
    result = benchmark.evaluate(repeated_samples, DATA, iterations=100)
    assert (result["samples"], result["prompts"], result["targets"], result["replicates"]) == (
        128,
        64,
        29,
        2,
    )
    for metric in result["confidence"]["metrics"].values():
        assert metric == {"mean": 1.0, "low": 1.0, "high": 1.0}


def test_every_replicate_must_cover_the_same_fixed_prompts(repeated_samples):
    with pytest.raises(ValueError, match="missing 1 prompts"):
        benchmark.evaluate(repeated_samples[:-1], DATA, iterations=10)


def test_reused_sampling_noise_is_rejected(repeated_samples):
    repeated_samples[-1]["sampling_seed"] = repeated_samples[0]["sampling_seed"]
    try:
        with pytest.raises(ValueError, match="sampling seed .* repeated"):
            benchmark.evaluate(repeated_samples, DATA, iterations=10)
    finally:
        repeated_samples[-1]["sampling_seed"] += len(repeated_samples) - 1


def test_the_cli_writes_the_full_machine_readable_report(repeated_samples, tmp_path, capsys):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps(row) + "\n" for row in repeated_samples))
    report = tmp_path / "benchmark.json"

    assert (
        main(
            [
                "benchmark",
                str(samples),
                "--bootstrap",
                "20",
                "--json",
                str(report),
            ]
        )
        == 0
    )
    payload = json.loads(report.read_text())
    assert payload["confidence"]["level"] == 0.95
    assert len(payload["by_target"]) == 29
    assert "MODULE BENCHMARK" in capsys.readouterr().out
