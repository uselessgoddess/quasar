"""The fixed module benchmark: stable prompts, explicit strata, no train leak.

The mixed held-out prompt stream is useful as a broad regression test, but it
cannot answer whether the model learned modules: a 48-prompt prefix contains
only about five of them.  This module builds the primary evaluation set
directly from the planner catalogue instead:

* all 29 targets in the pinned pre-DAG baseline appear at least twice;
* every branching target gets a third prompt;
* the electronic-circuit flagship gets the remaining 64th slot;
* each prompt's seed is derived from its semantic identity, so adding or
  reordering a target does not reshuffle the prompts that already exist.

The reference blueprints are not training examples.  :mod:`dataset` reserves
their canonical layouts before writing either shard, which gives the benchmark
the same design-level holdout guarantee as the ordinary validation split.

``dag-v1`` is the companion measurement for the first multi-belt recipe graph:
32 prompts over eight held-out green-science route combinations.  It stays
separate so the pinned module baseline does not move when DAG data changes.
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib
import random
from dataclasses import dataclass
from typing import Any

from . import augment, grammar, plan, synth, validate
from .blueprint import Blueprint
from .grammar import Spec
from .prototypes import Data, load

VERSION = "module-v1"
DAG_VERSION = "dag-v1"
DEFAULT_SIZE = 64
DAG_SIZE = 32
DAG_VARIANTS = 4
DEFAULT_SEED = 19
DAG_TARGET = "logistic-science-pack|iron-plate,copper-plate|d4|factory"

# The benchmark is a versioned measurement, not a view of whatever happens to
# be in the current generator catalogue.  Keep the exact issue-19 baseline
# here so adding diamond DAGs or green science later cannot silently change the
# question being asked by ``module-v1``.
TARGETS = (
    "arithmetic-combinator|iron-plate,copper-cable|d2|stack",
    "automation-science-pack|copper-plate,iron-plate|d2|stack",
    "boiler|stone,iron-plate|d2|fork",
    "burner-inserter|iron-plate|d2|stack",
    "constant-combinator|iron-plate,copper-cable|d2|stack",
    "decider-combinator|iron-plate,copper-cable|d2|stack",
    "destroyer-capsule|defender-capsule,advanced-circuit,electronic-circuit|d2|fork",
    "discharge-defense-remote|iron-plate,copper-cable|d2|stack",
    "electric-energy-interface|iron-plate,copper-cable|d2|stack",
    "electronic-circuit|iron-plate,copper-plate|d2|stack",
    "fast-transport-belt|iron-plate,iron-gear-wheel|d2|stack",
    "fast-underground-belt|iron-plate,transport-belt|d2|fork",
    "flamethrower|steel-plate,iron-plate|d2|stack",
    "green-wire|iron-plate,copper-cable|d2|stack",
    "pipe-to-ground|iron-plate|d2|stack",
    "rail-chain-signal|iron-plate,copper-cable|d2|stack",
    "rail-signal|iron-plate,copper-cable|d2|stack",
    "red-wire|iron-plate,copper-cable|d2|stack",
    "repair-pack|iron-plate,copper-cable|d2|fork",
    "small-electric-pole|wood,copper-plate|d2|stack",
    "transport-belt|iron-plate|d2|stack",
    "underground-belt|iron-plate,iron-gear-wheel|d2|stack",
    "discharge-defense-remote|iron-plate,copper-plate|d3|stack",
    "electric-energy-interface|iron-plate,copper-plate|d3|stack",
    "fast-underground-belt|iron-plate,iron-gear-wheel|d3|stack",
    "rail-chain-signal|iron-plate,copper-plate|d3|stack",
    "rail-signal|iron-plate,copper-plate|d3|stack",
    "repair-pack|iron-plate,copper-plate|d3|fork",
    "underground-belt|iron-plate|d3|stack",
)

# A fractional holdout over the four geometry axes.  Every individual value is
# present in training, but these eight combinations are not: generalisation is
# measured without reserving all 32 forms and thereby removing the DAG from the
# corpus altogether.
DAG_FORMS = tuple(
    form
    for form in synth.FACTORY_FORMS
    if form.upstream_swapped == bool(form.spacing % 2) and form.middle_swapped == bool(form.margin)
)


@dataclass(frozen=True)
class Case:
    """One fixed prompt and the working reference it was measured from."""

    prompt_id: str
    target_id: str
    variant: int
    shape: str
    blueprint: Blueprint
    spec: Spec
    reference: str
    benchmark: str = VERSION
    layout: str | None = None

    def record(self) -> dict:
        """The JSON record consumed by ``quasar generate``."""
        record = {
            "benchmark": self.benchmark,
            "benchmark_prompt": self.prompt_id,
            "benchmark_target": self.target_id,
            "benchmark_variant": self.variant,
            "kind": "module",
            "shape": self.shape,
            "spec": self.spec.to_dict(),
            "prompt": grammar.prompt(self.spec),
            "reference": self.reference,
        }
        if self.layout is not None:
            record["benchmark_layout"] = self.layout
        return record


def target_id(target: plan.Module) -> str:
    """A stable stratum name that distinguishes depth and declared supply."""
    supply = ",".join(target.supply)
    return f"{target.product}|{supply}|d{target.depth}|{target.shape}"


@functools.cache
def cases(
    data: Data | None = None,
    *,
    size: int = DEFAULT_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[Case, ...]:
    """Build the versioned, stratified prompt set.

    Allocation is deterministic and explicit.  Every target gets the quotient
    of ``size / targets``.  Remaining slots go to forks first, then the
    electronic-circuit flagship, then catalogue order.  At the current 29
    targets and size 64 that is 2 each + 5 forks + 1 flagship.
    """
    data = data or load()
    available = {target_id(target): target for target in plan.modules(data)}
    missing = [name for name in TARGETS if name not in available]
    if missing:
        raise ValueError(f"{VERSION} targets are no longer available: {', '.join(missing)}")
    targets = [available[name] for name in TARGETS]
    if size < len(targets):
        raise ValueError(f"benchmark size {size} cannot cover {len(targets)} module targets")

    per_target = {target: size // len(targets) for target in targets}
    remaining = size - sum(per_target.values())
    priority = [target for target in targets if target.shape == "fork"]
    priority += [
        target
        for target in targets
        if target.product == "electronic-circuit" and target not in priority
    ]
    priority += [target for target in targets if target not in priority]
    for target in priority[:remaining]:
        per_target[target] += 1

    found: list[Case] = []
    keys: set[str] = set()
    prompts: set[str] = set()
    for target in targets:
        name = target_id(target)
        for variant in range(per_target[target]):
            # A collision is unlikely but must not silently reduce a fixed
            # benchmark.  Retrying only this semantic case leaves every other
            # prompt byte-identical.
            for attempt in range(100):
                draw_seed = _seed(f"{VERSION}:{seed}:{name}:{variant}:{attempt}")
                rng = random.Random(draw_seed)
                blueprint, spec = synth.module_for(rng, data, target)
                reference = grammar.serialise(blueprint, data, spec)
                key = augment.canonical(blueprint, data)
                prompt = grammar.prompt(spec)
                if key in keys or prompt in prompts:
                    continue
                report = validate.grade(reference, data)
                if not (
                    report.valid
                    and report.delivers == report.fed == report.working == 1.0
                    and report.mixed == report.leaks == 0
                ):
                    raise ValueError(f"benchmark reference is not a working module: {name}")
                keys.add(key)
                prompts.add(prompt)
                found.append(
                    Case(
                        prompt_id=f"{name}|v{variant}",
                        target_id=name,
                        variant=variant,
                        shape=target.shape,
                        blueprint=blueprint,
                        spec=spec,
                        reference=reference,
                    )
                )
                break
            else:  # pragma: no cover - a catalogue with fewer forms is a hard error
                raise ValueError(f"could not draw a unique benchmark prompt for {name} v{variant}")
    return tuple(found)


@functools.cache
def dag_cases(
    data: Data | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[Case, ...]:
    """Build fixed prompts over held-out green-science route combinations.

    ``module-v1`` remains pinned to its original 29 pre-DAG targets.  This
    companion benchmark asks the narrower question introduced by the new
    training data: can the model connect the same diamond recipe graph in eight
    held-out combinations of sibling order, spacing, and edge margin? Four
    port/orientation prompts per layout make 32 fixed conditions.
    """
    data = data or load()
    available = {target_id(target): target for target in plan.modules(data)}
    if DAG_TARGET not in available:
        raise ValueError(f"{DAG_VERSION} target is no longer available: {DAG_TARGET}")
    target = available[DAG_TARGET]
    name = target_id(target)
    found: list[Case] = []
    keys: set[str] = set()
    prompts: set[str] = set()
    for form in DAG_FORMS:
        layout_key: str | None = None
        for variant in range(DAG_VARIANTS):
            for attempt in range(100):
                draw_seed = _seed(f"{DAG_VERSION}:{seed}:{name}:{form.name}:{variant}:{attempt}")
                rng = random.Random(draw_seed)
                blueprint, spec = synth.module_for(rng, data, target, factory_form=form)
                reference = grammar.serialise(blueprint, data, spec)
                key = augment.canonical(blueprint, data)
                prompt = grammar.prompt(spec)
                if prompt in prompts:
                    continue
                if layout_key is None:
                    if key in keys:
                        continue
                    keys.add(key)
                    layout_key = key
                elif key != layout_key:
                    raise ValueError(f"{form.name} is not one canonical layout")
                report = validate.grade(reference, data)
                if not (
                    report.valid
                    and report.delivers == report.fed == report.working == 1.0
                    and report.mixed == report.leaks == 0
                ):
                    raise ValueError(f"benchmark reference is not a working DAG: {form.name}")
                prompts.add(prompt)
                found.append(
                    Case(
                        prompt_id=f"{name}|{form.name}|v{variant}",
                        target_id=name,
                        variant=variant,
                        shape=target.shape,
                        blueprint=blueprint,
                        spec=spec,
                        reference=reference,
                        benchmark=DAG_VERSION,
                        layout=form.name,
                    )
                )
                break
            else:  # pragma: no cover - the catalogue no longer supports the measurement
                raise ValueError(f"could not draw a unique DAG prompt for {form.name} v{variant}")
    if len(found) != DAG_SIZE:
        raise ValueError(f"{DAG_VERSION} expected {DAG_SIZE} prompts, found {len(found)}")
    return tuple(found)


def write(path: pathlib.Path, selected: tuple[Case, ...]) -> None:
    """Write prompts as JSONL in their stable catalogue/variant order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case.record(), sort_keys=True) for case in selected]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def reserved(selected: tuple[Case, ...], data: Data) -> set[str]:
    """Canonical layouts that must stay out of train and validation shards."""
    return {augment.canonical(case.blueprint, data) for case in selected}


METRICS = {
    "parses": "parse_rate",
    "valid": "valid_rate",
    "spec": "spec_rate",
    "quality": "mean_quality",
    "delivers": "mean_delivers",
    "fed": "mean_fed",
    "working": "mean_working",
    "flow": "mean_flow",
}


def evaluate(
    samples: list[dict],
    data: Data | None = None,
    *,
    confidence: float = 0.95,
    iterations: int = 2_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Grade a complete repeated benchmark and estimate uncertainty.

    The bootstrap preserves the benchmark's target strata.  It resamples prompt
    variants within each target and sampling replicates within each prompt,
    rather than pretending 192 generations of 64 repeated specifications are
    192 unrelated observations.
    """
    data = data or load()
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be at least 1")
    if not samples:
        raise ValueError("no benchmark samples")

    versions = {sample.get("benchmark") for sample in samples}
    if len(versions) != 1:
        raise ValueError(f"expected one benchmark version, found {sorted(map(str, versions))}")
    version = next(iter(versions))
    if version not in {VERSION, DAG_VERSION}:
        raise ValueError(f"unknown benchmark {version}")

    required = ("benchmark_prompt", "benchmark_target", "shape", "replicate", "sampling_seed")
    for index, sample in enumerate(samples, 1):
        missing = [field for field in required if field not in sample]
        if missing:
            raise ValueError(f"sample {index} is missing {', '.join(missing)}")

    by_replicate: dict[int, list[tuple[dict, validate.Report]]] = {}
    all_rows = []
    for sample in samples:
        row = (sample, validate.grade(sample["text"], data))
        all_rows.append(row)
        by_replicate.setdefault(int(sample["replicate"]), []).append(row)

    selected_cases = cases(data) if version == VERSION else dag_cases(data)
    definitions = {case.prompt_id: case.record() for case in selected_cases}
    expected = set(definitions)
    found = {sample["benchmark_prompt"] for sample, _ in all_rows}
    missing, unknown = expected - found, found - expected
    if missing or unknown:
        raise ValueError(
            f"{version} prompt set differs: {len(missing)} missing, {len(unknown)} unknown"
        )
    seeds: set[int] = set()
    for sample, _ in all_rows:
        definition = definitions[sample["benchmark_prompt"]]
        fields = ["benchmark_target", "shape", "prompt"]
        if "benchmark_layout" in definition:
            fields.append("benchmark_layout")
        for field in fields:
            if sample.get(field) != definition[field]:
                raise ValueError(f"{sample['benchmark_prompt']} has changed {field}")
        sample_seed = int(sample["sampling_seed"])
        if sample_seed in seeds:
            raise ValueError(f"sampling seed {sample_seed} is repeated")
        seeds.add(sample_seed)

    for replicate, rows in by_replicate.items():
        prompt_ids = [sample["benchmark_prompt"] for sample, _ in rows]
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError(f"replicate {replicate} repeats a benchmark prompt")
        missing = expected - set(prompt_ids)
        if missing:
            raise ValueError(f"replicate {replicate} is missing {len(missing)} prompts")

    reports = [report for _, report in all_rows]
    summary = validate.summarise(reports)
    intervals = _bootstrap(all_rows, confidence=confidence, iterations=iterations, seed=seed)

    replicas = []
    for replicate, rows in sorted(by_replicate.items()):
        seeds = [int(sample["sampling_seed"]) for sample, _ in rows]
        replicas.append(
            {
                "replicate": replicate,
                "sampling_seed_min": min(seeds),
                "sampling_seed_max": max(seeds),
                "summary": validate.summarise([report for _, report in rows]).to_dict(),
            }
        )

    shapes = {}
    for shape in sorted({sample["shape"] for sample, _ in all_rows}):
        selected = [report for sample, report in all_rows if sample["shape"] == shape]
        shapes[shape] = validate.summarise(selected).to_dict()

    targets = {}
    for name in sorted({sample["benchmark_target"] for sample, _ in all_rows}):
        selected = [report for sample, report in all_rows if sample["benchmark_target"] == name]
        targets[name] = validate.summarise(selected).to_dict()

    layouts = {}
    for layout in sorted(
        {sample["benchmark_layout"] for sample, _ in all_rows if "benchmark_layout" in sample}
    ):
        selected = [
            report for sample, report in all_rows if sample.get("benchmark_layout") == layout
        ]
        layouts[layout] = validate.summarise(selected).to_dict()

    return {
        "benchmark": version,
        "samples": len(samples),
        "prompts": len(expected),
        "targets": len(targets),
        "fork_prompts": len(
            {sample["benchmark_prompt"] for sample, _ in all_rows if sample["shape"] == "fork"}
        ),
        "factory_prompts": len(
            {sample["benchmark_prompt"] for sample, _ in all_rows if sample["shape"] == "factory"}
        ),
        "replicates": len(by_replicate),
        "summary": summary.to_dict(),
        "confidence": {
            "level": confidence,
            "method": "stratified prompt-and-replicate bootstrap",
            "iterations": iterations,
            "metrics": intervals,
        },
        "by_replicate": replicas,
        "by_shape": shapes,
        "by_target": targets,
        "by_layout": layouts,
    }


def _bootstrap(rows, *, confidence: float, iterations: int, seed: int) -> dict[str, dict]:
    strata: dict[str, dict[str, list[validate.Report]]] = {}
    for sample, report in rows:
        target = str(sample["benchmark_target"])
        prompt = str(sample["benchmark_prompt"])
        strata.setdefault(target, {}).setdefault(prompt, []).append(report)

    rng = random.Random(seed)
    draws = {name: [] for name in METRICS}
    for _ in range(iterations):
        selected = []
        for prompts in strata.values():
            groups = list(prompts.values())
            for _ in range(len(groups)):
                reports = rng.choice(groups)
                selected.extend(rng.choice(reports) for _ in range(len(reports)))
        summary = validate.summarise(selected)
        for name, attribute in METRICS.items():
            draws[name].append(getattr(summary, attribute))

    alpha = (1.0 - confidence) / 2.0
    point = validate.summarise([report for _, report in rows])
    return {
        name: {
            "mean": getattr(point, attribute),
            "low": _quantile(draws[name], alpha),
            "high": _quantile(draws[name], 1.0 - alpha),
        }
        for name, attribute in METRICS.items()
    }


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round(probability * (len(ordered) - 1))
    return ordered[max(0, min(index, len(ordered) - 1))]


def _seed(identity: str) -> int:
    """A process- and platform-independent integer seed."""
    return int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "little")
