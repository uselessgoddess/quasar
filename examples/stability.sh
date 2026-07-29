#!/usr/bin/env bash
# The recipe from issue #23, on the card that reported it, in every precision.
#
# The report was three runs of `train tiny-turbo --micro-batch 4 --accum 32
# --ssd serial --checkpointing false --muon true`, and all three ended before
# step 90: fp16 on all three seams at step 89 and fp16 on the head alone at
# step 69 with "non-finite gradients ... minimum loss scale 1", fp32 at step 80
# with "164 tensors are not finite, from embed.weight through norm.gamma".
#
# So this runs exactly that recipe, past all three of those steps, on all four
# precision arms. A unit test can show the guard rejects a NaN it was handed;
# only the card can show whether the recipe produces one. The arms are
# deliberately the same run with one flag moved, because the question the issue
# asks is which precision the failure belongs to.
#
#     examples/stability.sh                      # CPU, whatever backend is default
#     BACKEND=vulkan examples/stability.sh       # the card the issue reports
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/stability}
work=${STABILITY_WORK:-$repo_root/$log_dir/work}
# Well past step 89, which is the latest of the three reported failures. The
# reported run was 4000 steps; the failures are all inside the first hundred,
# so the first hundred is what has to be reproduced to say anything about them.
steps=${STABILITY_STEPS:-120}
# The reported accumulation, and the knob CI turns down. Four verbatim arms are
# 63M tokens and over an hour on the one card the whole self-hosted half of CI
# shares. A quarter of the accumulation is not a gentler test: `--lr 3e-3` is
# tuned against the full 131k-token batch, so a quarter of it is four times the
# gradient noise at the same rate — strictly more likely to produce the spike
# being reproduced. What it does not change is `--micro-batch 4`, which is the
# whole of the memory shape and therefore of whether the card holds the run.
accum=${STABILITY_ACCUM:-32}
# 120,000 documents are 12.9M tokens against the 15.7M one verbatim arm reads
# (4 x 32 x 1024 x 120), so the loader wraps once — far less than the reported
# 4000-step run wraps over any corpus this box can hold.
docs=${STABILITY_DOCS:-120000}
backend=${BACKEND:-}
mkdir -p "$repo_root/$log_dir" "$work"
cd "$repo_root"

# One feature set for the corpus stages as well as the training arms: they are
# the same binary, and building it twice would spend more wall clock on rustc
# than the four arms spend on the card.
features=()
[ -n "$backend" ] && features=(--no-default-features --features "$backend")
quasar=(cargo run --release -q "${features[@]}" --)

# `train tiny-turbo` reads its vocabulary from the shards, not from the preset,
# so the corpus has to be fitted at the preset's own vocabulary or the run is a
# different model. 8192 is `config::SMALL_VOCAB`.
vocab=${STABILITY_VOCAB:-8192}

if [[ ! -f "$work/shards/train/meta.json" ]]; then
    python3 - "$work/corpus.jsonl" "$docs" <<'PY'
import itertools
import json
import random
import sys

# Zipfian nonsense over a lexicon wide enough that 8192 BPE merges have
# something to merge. The width is not decoration: `prepare` writes the fitted
# vocabulary into the shards and `train` reads the model's from there, so a
# lexicon of a few hundred words would quietly train a few-hundred-token model
# and reproduce none of the head GEMM, logits or embedding the report ran.
#
# Structure the model can partly learn and partly cannot is what makes a loss
# curve worth reading; a corpus it memorises in fifty steps would hide a
# divergence rather than provoke one. So: a Zipfian lexicon of random syllable
# strings, and sentences drawn from it independently.
random.seed(23)
syllables = [c + v for c in "bcdfgklmnprstvz" for v in "aeiou"]
lexicon = sorted({
    "".join(random.choices(syllables, k=random.randint(2, 4))) for _ in range(60_000)
})
random.shuffle(lexicon)
# Accumulated once, not per call: `random.choices` builds this itself otherwise,
# and 120,000 cumulative sums over a 60,000-word lexicon is eight minutes of
# corpus generation in front of a GPU job that runs for twenty.
cum = list(itertools.accumulate(1.0 / (index + 1) for index in range(len(lexicon))))
with open(sys.argv[1], "w") as out:
    for _ in range(int(sys.argv[2])):
        words = random.choices(lexicon, cum_weights=cum, k=random.randint(40, 120))
        out.write(json.dumps({"text": " ".join(words)}) + "\n")
PY

    "${quasar[@]}" tokenizer "$work/corpus.jsonl" \
        --out "$work/tokenizer.json" --vocab-size "$vocab"
    "${quasar[@]}" prepare "$work/corpus.jsonl" \
        --tokenizer "$work/tokenizer.json" --out "$work/shards"
fi

# The reported command, verbatim apart from the length and the decay.
#
# The decay has to be given because the schedule is WSD and the default is 2500
# steps, which does not fit inside 120. One is the faithful choice rather than a
# proportional 20%: the reported run holds its peak rate flat from step 20 to
# step 1500, so every step of this reproduction is a peak-rate step there too,
# and annealing a short run would anneal away exactly the instability it is here
# to provoke.
common=(
    train tiny-turbo
    --data "$work/shards"
    --micro-batch 4
    --accum "$accum"
    --ssd serial
    --checkpointing false
    --muon true
    --steps "$steps"
    --warmup 20
    --decay 1
    --log-every 1
    --eval-every 0
    --save-every 0
)

run_arm() {
    local label=$1
    shift
    local status=0

    set +e
    # Parameter norms every ten steps. A run that comes apart is exactly the
    # thing a log cannot explain afterwards, and this is the run whose
    # explanation the issue is asking for.
    QUASAR_TRACE_NORMS=10 "${quasar[@]}" \
        "${common[@]}" --out "$work/$label" "$@" \
        2>&1 | tee "$log_dir/$label.log"
    status=${PIPESTATUS[0]}
    set -e

    if ((status != 0)); then
        echo "stability arm=$label exited=$status" > "$log_dir/$label.result"
        return "$status"
    fi

    # A run that ends is not the same as a run that trained. Both halves are
    # checked: the loss has to be a number, and it has to be below the uniform
    # baseline it started at, or the arm survived by learning nothing.
    awk -v label="$label" -v vocab="$vocab" '
        /^final: / {
            for (i = 1; i <= NF; i++) {
                if ($i == "loss") text = $(i + 1)
            }
            seen = 1
        }
        /^non-finite gradient/ { skipped++ }
        END {
            chance = log(vocab)
            printf "stability arm=%s final_loss=%s chance=%.4f skipped_steps=%d\n",
                label, seen ? text : "none", chance, skipped
            if (!seen) {
                print "the run printed no final loss" > "/dev/stderr"
                exit 1
            }
            # On the string rather than on `text + 0`: Rust prints a diverged
            # loss as `NaN`, and what that becomes in arithmetic is the awk
            # implementations arguing — gawk says nan, others say zero, and zero
            # would pass a gate meant to catch exactly this.
            if (text !~ /^-?[0-9]+(\.[0-9]+)?$/ || text + 0 >= chance) {
                print "the run ended without learning" > "/dev/stderr"
                exit 1
            }
        }
    ' "$log_dir/$label.log" > "$log_dir/$label.result"
}

# Every arm runs even after one fails. Which precisions come apart and which
# survive is the whole question, and a script that stops at the first failure
# answers it for one arm and leaves the other three to a second CI run on a
# queue of one card.
failed=()
arm() {
    local label=$1
    shift
    run_arm "$label" "$@" || failed+=("$label")
}

# fp32 first: it is the arm the issue reports as diverging with no reduced
# precision anywhere in it, which is what says the failure is not fp16's.
arm fp32 --head-dtype fp32 --ffn-dtype fp32 --mamba-dtype fp32
arm f16-head --head-dtype f16 --ffn-dtype fp32 --mamba-dtype fp32
arm f16-all --head-dtype f16 --ffn-dtype f16 --mamba-dtype f16
arm bf16-all --head-dtype bf16 --ffn-dtype bf16 --mamba-dtype bf16

echo "== stability =="
cat "$log_dir"/*.result
if ((${#failed[@]})); then
    echo "arms that did not train: ${failed[*]}" >&2
    exit 1
fi

cat "$log_dir"/*.result
