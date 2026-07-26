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
# 6,000 designs is roughly where the generators start repeating themselves at
# four variants each; see factorio/experiments/saturation.py.
designs=${DESIGNS:-6000}
# Empty means the preset's own length; see the train stage.
steps=${STEPS:-}
# `nano` samples one token per forward pass with no cache, so this is the knob
# that decides whether the run fits its budget, not the training length.
prompts=${PROMPTS:-24}
backend=${BACKEND:-}
# Human blueprints, if a cache has been fetched — the generators saturate around
# 6,300 layouts and the Chinchilla budget wants more than that. Absent, the run
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
for checkpoint in "$out"/step_*; do
    step=$(basename "$checkpoint" | tr -dc '0-9')
    "${quasar[@]}" generate "$checkpoint" \
        --tokenizer "$corpus/tokenizer.json" \
        --prompts "$out/prompts.jsonl" --out "$out/samples-$step.jsonl" \
        --tokens 460 --temperature 0.7 --top-k 20
done

stage grade
last=$(ls "$out"/samples-*.jsonl | tail -1)
"${harness[@]}" grade "$last" \
    --sheet "$out/sheet.png" --json "$out/grade.json" --columns 4
"${harness[@]}" grade "$last" --sheet "$out/failures.png" --order worst >/dev/null

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
"${harness[@]}" plot "$out/train.log" "$out/metrics.png" --grade "$out"/samples-*.jsonl

# The best generation, as a blueprint string. This is the whole point of the
# exercise: something that can be pasted into the game.
"$python" - "$last" "$out/best.txt" <<'PY'
import json
import pathlib
import sys

sys.path.insert(0, "factorio/src")
from quasar_factorio import prototypes, validate  # noqa: E402

data = prototypes.load()
samples = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines() if line]


def rank(sample):
    # Flow first, then quality, then entity count. Quality alone was the whole
    # ranking until the run analysed in docs/FACTORIO.md finished with a dozen
    # generations at a flat 1.0 -- at which point the tie-break is doing all the
    # work and "best" means "biggest". Flow asks whether the thing would make
    # anything, which is a question the ceiling has not been reached on.
    report = validate.grade(sample["text"], data)
    return report.flows(), report.quality(), report.entities


ranked = sorted(samples, key=rank, reverse=True)
pathlib.Path(sys.argv[2]).write_text(ranked[0]["text"] if ranked else "")
PY
"${harness[@]}" render "$out/best.txt" "$out/best.png" || true
"${harness[@]}" export "$out/best.txt" >"$out/best.blueprint" || true

echo
timings[-1]="${timings[-1]}$((SECONDS - started))s"
printf '%s\n' "${timings[@]}"
printf '%-24s%ss\n' total "$SECONDS"
echo
echo "artifacts in $out: metrics.png sheet.png failures.png corpus.png best.png frames/"
