#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

sampler_pid=""
if command -v rocm-smi >/dev/null 2>&1; then
    (
        while true; do
            date -u +'%Y-%m-%dT%H:%M:%SZ'
            rocm-smi --showmeminfo vram --showuse --showclocks --showtemp --showpower
            sleep 1
        done
    ) >"$log_dir/vram-baseline.log" 2>&1 &
    sampler_pid=$!
fi

set +e
cargo run --release --no-default-features --features vulkan \
    --example train_bench -- \
    --model tiny-turbo --micro-batch 4 --accum 32 \
    --warmup 1 --steps 3 --max-steps 9 --dtype f32 --ssd serial \
    --checkpointing false --muon true --head-dtype f16 \
    2>&1 | tee "$log_dir/baseline.log"
benchmark_status=${PIPESTATUS[0]}
set -e

if [[ -n $sampler_pid ]]; then
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    peak_bytes=$(
        awk '
            /GPU\[0\].*VRAM Total Used Memory/ && $NF + 0 > peak {
                peak = $NF + 0
            }
            END { printf "%.0f", peak }
        ' "$log_dir/vram-baseline.log"
    )
    peak_gib=$(awk -v peak="$peak_bytes" 'BEGIN { printf "%.3f", peak / 1073741824 }')
    printf 'vram peak_bytes=%s peak_gib=%s limit_gib=15.000\n' "$peak_bytes" "$peak_gib" \
        | tee "$log_dir/vram-baseline.result"
    if ((peak_bytes <= 0)); then
        echo "VRAM sampler did not report GPU[0] usage" >&2
        benchmark_status=1
    elif ((peak_bytes > 15 * 1024 * 1024 * 1024)); then
        echo "peak VRAM exceeds the 15 GiB production gate" >&2
        benchmark_status=1
    fi
fi

exit "$benchmark_status"
