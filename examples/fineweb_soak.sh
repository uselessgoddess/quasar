#!/usr/bin/env bash
# One production-shaped, multi-hour training run on a pinned FineWeb-Edu 10BT
# shard. This is deliberately not a benchmark sweep: it answers whether the
# fastest measured recipe keeps learning after the short stability gate ends.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/fineweb}
data_root=${FINEWEB_DATA:-$repo_root/data/fineweb-soak}
run_root=${FINEWEB_RUN:-$repo_root/$log_dir/run}
steps=${FINEWEB_STEPS:-1800}
minimum_throughput=${FINEWEB_MIN_THROUGHPUT:-17000}
tokenizer_docs=${FINEWEB_TOKENIZER_DOCS:-200000}

# Pin the corpus, not only its moving branch name. The first of fourteen 10BT
# shards is 2.15 GB compressed and contains much more than this run's 235.9M
# tokens, so downloading the other 26 GB would change no batch the test sees.
dataset_revision=87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
dataset_name=000_00000.parquet
dataset_bytes=2152819114
dataset_sha256=b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
dataset_url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/$dataset_revision/sample/10BT/$dataset_name?download=true"
dataset_file=$data_root/$dataset_name

mkdir -p "$repo_root/$log_dir" "$data_root" "$run_root"
cd "$repo_root"

download_dataset() {
    local partial=$dataset_file.partial
    if [[ -f $dataset_file ]]; then
        local existing
        existing=$(stat -c %s "$dataset_file")
        if ((existing == dataset_bytes)); then
            return
        fi
        echo "$dataset_file has $existing bytes, expected $dataset_bytes" >&2
        return 1
    fi

    echo "downloading FineWeb-Edu 10BT shard at $dataset_revision"
    curl --fail --location --retry 5 --retry-all-errors --continue-at - \
        --output "$partial" "$dataset_url"
    local received
    received=$(stat -c %s "$partial")
    if ((received != dataset_bytes)); then
        echo "downloaded $received bytes, expected $dataset_bytes" >&2
        return 1
    fi
    mv "$partial" "$dataset_file"
}

download_dataset
printf '%s  %s\n' "$dataset_sha256" "$dataset_file" | sha256sum --check -

# Use one Vulkan build for fitting, preparation, and training. The tokenizer is
# fitted on real FineWeb text; 200k documents are ample for 8192 BPE entries,
# while fitting all 10BT documents would spend GPU queue time on a CPU prelude.
quasar=(
    cargo run --release -q --no-default-features --features vulkan --
)
tokenizer=$data_root/tokenizer-8192.json
shards=$data_root/shards-8192
if [[ ! -f $tokenizer ]]; then
    "${quasar[@]}" tokenizer "$dataset_file" \
        --out "$tokenizer" --vocab-size 8192 --docs "$tokenizer_docs"
fi
if [[ ! -f $shards/train/meta.json || ! -f $shards/valid/meta.json ]]; then
    "${quasar[@]}" prepare "$dataset_file" --tokenizer "$tokenizer" --out "$shards"
fi

# No wrapping: every training token in the soak comes from a different window
# of the selected FineWeb shard. Fail before allocating the GPU if that ceases
# to be true after a dataset or recipe change.
required_tokens=$((steps * 4 * 32 * 1024))
train_tokens=$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' \
        "$shards/train/meta.json"
)
if ((train_tokens < required_tokens)); then
    echo "FineWeb shard has $train_tokens train tokens, need $required_tokens" >&2
    exit 1
fi
printf '%s\n' \
    "dataset=HuggingFaceFW/fineweb-edu/sample/10BT/$dataset_name" \
    "dataset_revision=$dataset_revision" \
    "dataset_sha256=$dataset_sha256" \
    "dataset_bytes=$dataset_bytes" \
    "train_tokens=$train_tokens" \
    "required_tokens=$required_tokens" \
    "tokenizer_docs=$tokenizer_docs" \
    >"$log_dir/dataset.result"

training_log=$log_dir/training.log
started=$(date +%s)
status=0
set +e
# These are the fastest accepted RX 9070 XT settings, written explicitly so a
# future preset edit cannot turn this evidence into a different recipe. A full
# 4x32 batch reduces optimizer and logging overhead; 1800 steps are 235.9M
# tokens, or about 3.85 hours at the already measured 17.08k tok/s. WSD spends
# most of that at the same 3e-3 peak that issue #23 originally lost before step
# 90, then anneals for the final fifth so the endpoint is meaningful.
QUASAR_TRACE_NORMS=${QUASAR_TRACE_NORMS:-200} "${quasar[@]}" \
    train tiny-turbo \
    --data "$shards" \
    --out "$run_root" \
    --steps "$steps" \
    --micro-batch 4 \
    --accum 32 \
    --lr 3e-3 \
    --warmup 20 \
    --decay 360 \
    --ssd serial \
    --checkpointing false \
    --muon true \
    --head-dtype f16 \
    --ffn-dtype f16 \
    --mamba-dtype f16 \
    --log-every 20 \
    --eval-every 300 \
    --save-every 0 \
    2>&1 | tee "$training_log"
status=${PIPESTATUS[0]}
set -e
elapsed=$(($(date +%s) - started))

report_status=0
python3 examples/fineweb_report.py "$training_log" \
    --elapsed-seconds "$elapsed" \
    --expected-steps "$steps" \
    --minimum-throughput "$minimum_throughput" \
    --out "$log_dir/training.result" \
    || report_status=$?

if ((status != 0)); then
    exit "$status"
fi
exit "$report_status"
