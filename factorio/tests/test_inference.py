"""Constrained-versus-rejection experiment accounting."""

import random

from quasar_factorio import benchmark, grammar, inference, prototypes, synth

DATA = prototypes.load()


def sample(seed=0):
    blueprint, spec = synth.module(random.Random(seed), DATA)
    text = grammar.serialise(blueprint, DATA, spec)
    return {
        "benchmark_prompt": f"prompt-{seed}",
        "prompt": grammar.prompt(spec),
        "text": text,
        "sampling_seed": seed,
    }


def test_rejection_keeps_the_first_hard_valid_attempt_and_records_real_compute():
    good = sample()
    broken = {**good, "text": good["prompt"] + " <e> nowhere x00 y00 d0 </bp>"}
    attempts = [
        {**broken, "replicate": 0, "sampling_seed": 10},
        {**good, "replicate": 1, "sampling_seed": 11},
        {**good, "replicate": 2, "sampling_seed": 12},
    ]
    chosen = inference.select(attempts, DATA)
    assert len(chosen) == 1
    assert chosen[0]["text"] == good["text"]
    assert chosen[0]["accepted"] is True
    assert chosen[0]["attempts_used"] == 2
    assert chosen[0]["attempt_budget"] == 3
    assert chosen[0]["replicate"] == 0


def test_rejection_returns_the_last_failure_when_the_budget_is_exhausted():
    good = sample()
    broken = {**good, "text": good["prompt"] + " </bp>"}
    chosen = inference.select(
        [
            {**broken, "replicate": 0, "sampling_seed": 10},
            {**broken, "replicate": 1, "sampling_seed": 11},
        ],
        DATA,
    )
    assert chosen[0]["accepted"] is False
    assert chosen[0]["attempts_used"] == chosen[0]["attempt_budget"] == 2


def test_comparison_uses_the_same_complete_fixed_benchmark_and_real_attempt_cost():
    constrained = []
    rejected = []
    for index, case in enumerate(benchmark.cases(DATA)):
        record = {
            **case.record(),
            "text": case.reference,
            "replicate": 0,
            "sampling_seed": index,
            "decoding": "constrained",
        }
        constrained.append(record)
        rejected.append(
            {
                **record,
                "decoding": "reject-regenerate",
                "attempt_budget": 2,
                "attempts_used": 1,
                "accepted": True,
            }
        )

    result = inference.compare(constrained, rejected, DATA, iterations=2)

    assert result["prompts"] == 64
    assert result["methods"]["constrained"]["generated_attempts"] == 64
    assert result["methods"]["reject_regenerate"]["generated_attempts"] == 128
    assert result["methods"]["constrained"]["summary"]["mean_working"] == 1.0
    assert result["methods"]["reject_regenerate"]["benchmark"]["benchmark"] == "module-v1"
