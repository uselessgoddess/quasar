#!/usr/bin/env bash
# The blueprint model end to end: corpus, training, generation, grade, plots.
#
# Written to fit inside an hour on one 16 GB card and to leave behind
# everything a person would want to look at afterwards — a preview of what went
# in, a contact sheet of what came out, and one board with every metric on it.
#
#     examples/factorio.sh                    # CPU, whatever backend is default
#     BACKEND=vulkan examples/factorio.sh     # the real thing
#     STEPS=2000 DESIGNS=20000 examples/factorio.sh runs/nano
#     REAL=factorio/data/blueprints.jsonl examples/factorio.sh
#
# Every knob is an environment variable because the defaults here are the CI
# defaults, and CI is the one caller that cannot afford to overrun.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

out=${1:-runs/nano}
corpus=${CORPUS:-$out/corpus}
# 6,000 draws is roughly where the generators start repeating themselves at
# four variants each; see factorio/experiments/saturation.py.
designs=${DESIGNS:-6000}
# Empty means the preset's own length; see the train stage.
steps=${STEPS:-}
# `nano` samples one token per forward pass with no cache, so this is the knob
# that decides whether the run fits its budget, not the training length.
prompts=${PROMPTS:-24}
# Two independent generations of every fixed prompt are the minimum useful
# estimate of sampling variance.  They share one checkpoint load, so repeats
# are much cheaper than separate generate invocations.
benchmark_repeats=${BENCHMARK_REPEATS:-2}
# The DAG curve needs coverage of all 24 fixed prompts (three targets × eight
# held-out routes) more than duplicate sampling. One draw per prompt keeps the
# extra pass inside the CI GPU budget.
dag_benchmark_repeats=${DAG_BENCHMARK_REPEATS:-1}
backend=${BACKEND:-}
# Human blueprints, if a cache has been fetched — 20,000 weighted draws are
# 14.4M tokens and the Chinchilla budget wants 71.4M. Absent, the run
# is synthetic-only and everything else about it is unchanged, which is what
# keeps this script runnable on a box with no network.
#
#     python factorio/tools/fetch_blueprints.py --count 6000
real=${REAL:-factorio/data/blueprints.jsonl}
# Designs, not records: the harvest stops as soon as it has this many, so it is
# the knob that decides how long the corpus stage takes.
real_limit=${REAL_LIMIT:-6000}

features=()
[ -n "$backend" ] && features=(--no-default-features --features "$backend")
quasar=(cargo run --release -q "${features[@]}" --)
# The harness has no dependencies, so any 3.11+ interpreter runs it; `PYTHON`
# is here for boxes whose `python3` is older than the tree.
python=${PYTHON:-python3}
harness=(env PYTHONPATH=factorio/src "$python" -m quasar_factorio.cli)

# Where the wall clock went, printed at the end. The budget is half an hour and
# the stages are wildly uneven, so a run that overruns should say which stage
# to shrink rather than leave it to be guessed at.
timings=()
stage() {
    [ ${#timings[@]} -eq 0 ] || timings[-1]="${timings[-1]}$((SECONDS - started))s"
    timings+=("$(printf '%-24s' "$1")")
    started=$SECONDS
    echo "== $1 $(printf '%.0s=' $(seq $((60 - ${#1}))))"
}

mkdir -p "$out"
stage corpus
mixture=()
if [ -f "$real" ]; then
    mixture=(--real "$real" --real-limit "$real_limit")
else
    echo "no blueprint cache at $real; synthetic corpus only" >&2
fi
"${harness[@]}" build "$corpus" --count "$designs" "${mixture[@]}"
"${harness[@]}" preview "$out/corpus.png" --corpus "$corpus" --count 12
"${harness[@]}" heatmap "$out/occupancy.png" --count 400

# The held-out prompts are specs the model was never trained on. Only the first
# few are sampled: generation is the expensive half of this script.
head -n "$prompts" "$corpus/prompts.jsonl" >"$out/prompts.jsonl"

stage train
# Unset `STEPS` means the preset decides, and what it decides is the Chinchilla
# budget: 20 tokens per parameter, about 4,300 steps of 16k tokens — see
# `cargo run -- budget factorio-nano --micro-batch 32`. A run shorter than that
# is not a smaller experiment, it is one whose curves have not separated the
# model from its initialisation, which is the whole reason this is the default.
#
# Setting STEPS derives the same proportions — 5% warmup, 20% decay — for the
# shorter run, which is what a smoke test wants.
schedule=()
if [ -n "$steps" ]; then
    warmup=$((steps / 20 > 0 ? steps / 20 : 1))
    decay=$((steps / 5 > 0 ? steps / 5 : 1))
    # A one- or two-step run — someone smoke-testing the script — asks for more
    # warmup and decay than it has steps. The stable phase is what gives way;
    # the schedule refuses the overlap rather than reinterpreting it.
    [ $((warmup + decay)) -le "$steps" ] || decay=$((steps - warmup))
    schedule=(
        --steps "$steps" --warmup "$warmup" --decay "$decay"
        --save-every "$((steps / 4 > 0 ? steps / 4 : 1))"
        --eval-every "$((steps / 20 > 0 ? steps / 20 : 1))"
        --log-every "$((steps / 60 > 0 ? steps / 60 : 1))"
    )
fi
# `tee` rather than `>`: the log is the input to the plots, and a run whose
# output cannot be watched is a run nobody will wait for.
"${quasar[@]}" train factorio-nano \
    --data "$corpus" --out "$out" "${schedule[@]}" \
    2>&1 | tee "$out/train.log"

stage generate
# Every checkpoint, not just the last one: the grade curve over training is the
# plot that says whether it is learning to build or learning to copy.
inference_checkpoint=
inference_samples=
for checkpoint in "$out"/step_*; do
    step=$(basename "$checkpoint" | tr -dc '0-9')
    "${quasar[@]}" generate "$checkpoint" \
        --tokenizer "$corpus/tokenizer.json" \
        --prompts "$out/prompts.jsonl" --out "$out/samples-$step.jsonl" \
        --tokens 460 --temperature 0.7 --top-k 20
    # This is the primary issue-19 measurement: the same versioned 64 prompts
    # at every checkpoint, stratified over all 29 module targets and the five
    # fork forms.  `--repeats` records a distinct seed on every output row.
    "${quasar[@]}" generate "$checkpoint" \
        --tokenizer "$corpus/tokenizer.json" \
        --prompts "$corpus/benchmark.jsonl" \
        --out "$out/benchmark-samples-$step.jsonl" \
        --tokens 460 --temperature 0.7 --top-k 20 \
        --repeats "$benchmark_repeats"
    # A separate held-out curve for three semantic DAGs: one prompt over each
    # of eight unseen route combinations per target. Keeping it out of
    # module-v1 preserves that benchmark's baseline; 24 balanced conditions
    # also fit below the previous 32-prompt generation budget.
    "${quasar[@]}" generate "$checkpoint" \
        --tokenizer "$corpus/tokenizer.json" \
        --prompts "$corpus/dag-benchmark.jsonl" \
        --out "$out/dag-samples-$step.jsonl" \
        --tokens 460 --temperature 0.7 --top-k 20 \
        --repeats "$dag_benchmark_repeats"
    # Pre-register the first checkpoint for inference A/B.  The full baseline
    # run measures it well below saturation; choosing it before looking at this
    # run's samples avoids selecting a conveniently favourable checkpoint.
    if [ -z "$inference_checkpoint" ]; then
        inference_checkpoint=$checkpoint
        inference_samples="$out/benchmark-samples-$step.jsonl"
    fi
done

stage grade
last=$(ls "$out"/samples-*.jsonl | tail -1)
benchmark_last=$(ls "$out"/benchmark-samples-*.jsonl | tail -1)
dag_last=$(ls "$out"/dag-samples-*.jsonl | tail -1)
"${harness[@]}" grade "$last" \
    --sheet "$out/sheet.png" --json "$out/grade.json" --columns 4
"${harness[@]}" grade "$benchmark_last" \
    --sheet "$out/failures.png" --order worst >/dev/null
"${harness[@]}" benchmark "$benchmark_last" --json "$out/benchmark.json"
"${harness[@]}" grade "$dag_last" \
    --sheet "$out/dag-failures.png" --order worst >/dev/null
"${harness[@]}" benchmark "$dag_last" --json "$out/dag-benchmark.json"

# One additional pass at the pre-registered, unsaturated first checkpoint
# compares prevention with post-hoc rejection on the same fixed prompts.
# Rejection uses its already generated unconstrained replicates, so its actual
# compute budget is counted rather than hidden behind the first accepted answer.
"${quasar[@]}" generate "$inference_checkpoint" \
    --tokenizer "$corpus/tokenizer.json" \
    --constraints "$corpus/constraints.json" \
    --prompts "$corpus/benchmark.jsonl" \
    --out "$out/constrained.jsonl" \
    --tokens 460 --temperature 0.7 --top-k 20
"${harness[@]}" select-rejections \
    "$inference_samples" "$out/rejected.jsonl"
"${harness[@]}" compare-inference \
    "$out/constrained.jsonl" "$out/rejected.jsonl" \
    --json "$out/inference.json"

# The module sheet is now the whole fixed benchmark, not a five-item accidental
# slice of the mixed prompts.  The mixed sheet above remains as a broad
# regression test for all generators.
"${harness[@]}" grade "$benchmark_last" --kind module \
    --sheet "$out/modules.png" --json "$out/grade-modules.json" --columns 4

# One frame per checkpoint, same prompt in the same cell every time, so the
# frames flip into a timelapse of the model learning to build:
#
#     ffmpeg -framerate 4 -pattern_type glob -i 'frames/*.png' training.mp4
mkdir -p "$out/frames"
for samples in "$out"/samples-*.jsonl; do
    step=$(basename "$samples" .jsonl | tr -dc '0-9')
    "${harness[@]}" grade "$samples" \
        --sheet "$out/frames/$step.png" --order given --columns 4 >/dev/null
done

stage plot
"${harness[@]}" plot "$out/train.log" "$out/metrics.png" \
    --grade "$out"/benchmark-samples-*.jsonl
"${harness[@]}" plot "$out/train.log" "$out/dag-metrics.png" \
    --kind factory --grade "$out"/dag-samples-*.jsonl

# The best generation, as a blueprint string. This is the whole point of the
# exercise: something that can be pasted into the game.
#
# Twice, and the second one is the interesting one: ranked over the whole
# mixture the winner is a belt lane often enough — dozens of samples tie at a
# perfect score and the tie-break is entity count — that `best.png` need never
# show the one task the run is about. `best-module.png` is the module the model
# built best, whatever the rest of the benchmark did.
"$python" - "$last" "$benchmark_last" "$dag_last" \
    "$out/best.txt" "$out/best-module.txt" "$out/best-dag.txt" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, "factorio/src")
from quasar_factorio import prototypes, validate  # noqa: E402

data = prototypes.load()
samples = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]
benchmark = [
    json.loads(line) for line in pathlib.Path(sys.argv[2]).read_text().splitlines() if line
]
dag = [json.loads(line) for line in pathlib.Path(sys.argv[3]).read_text().splitlines() if line]


def rank(sample):
    # Flow first, then quality, then entity count. Quality alone was the whole
    # ranking until the run analysed in docs/FACTORIO.md finished with a dozen
    # generations at a flat 1.0 -- at which point the tie-break is doing all the
    # work and "best" means "biggest". Flow asks whether the thing would make
    # anything, which is a question the ceiling has not been reached on.
    report = validate.grade(sample["text"], data)
    return report.flows(), report.quality(), report.entities


ranked = sorted(samples, key=rank, reverse=True)
pathlib.Path(sys.argv[4]).write_text(ranked[0]["text"] if ranked else "")

# Same ranking, over the dedicated module benchmark.
modules = sorted(benchmark, key=rank, reverse=True)
pathlib.Path(sys.argv[5]).write_text(modules[0]["text"] if modules else "")

# And over the held-out forms of both explicit recipe DAGs.
dags = sorted(dag, key=rank, reverse=True)
pathlib.Path(sys.argv[6]).write_text(dags[0]["text"] if dags else "")
PY
"${harness[@]}" render "$out/best.txt" "$out/best.png" || true
"${harness[@]}" export "$out/best.txt" >"$out/best.blueprint" || true
# `|| true` here is load-bearing rather than defensive: a run whose prompts held
# no module, or whose best module does not parse, leaves an empty file behind.
"${harness[@]}" render "$out/best-module.txt" "$out/best-module.png" || true
"${harness[@]}" export "$out/best-module.txt" >"$out/best-module.blueprint" || true
"${harness[@]}" render "$out/best-dag.txt" "$out/best-dag.png" || true
"${harness[@]}" export "$out/best-dag.txt" >"$out/best-dag.blueprint" || true

echo
timings[-1]="${timings[-1]}$((SECONDS - started))s"
printf '%s\n' "${timings[@]}"
printf '%-24s%ss\n' total "$SECONDS"
echo
echo "artifacts in $out: metrics.png dag-metrics.png sheet.png modules.png"
echo "                failures.png dag-failures.png corpus.png"
echo "                best.png best-module.png best-dag.png frames/"
