#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/rocm-rocwmma}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

common=(
    --rows 4096
    --input-features 640
    --output-features 32768
    --warmup 1
    --steps 3
    --max-steps 9
)

run_arm() {
    local label=$1
    local feature=$2
    local sampler_pid=""
    local status=0
    local vram_status=0

    if ! command -v rocm-smi >/dev/null 2>&1; then
        echo "rocm-smi is required for the 15 GiB VRAM gate" >&2
        return 1
    fi
    (
        while true; do
            date -u +'%Y-%m-%dT%H:%M:%SZ'
            rocm-smi --showmeminfo vram
            sleep 1
        done
    ) >"$log_dir/${label}-vram.log" 2>&1 &
    sampler_pid=$!

    set +e
    cargo run --release --no-default-features --features "$feature" \
        --example linear_backend_spike -- "${common[@]}" \
        2>&1 | tee "$log_dir/${label}.log"
    status=${PIPESTATUS[0]}
    set -e

    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true

    set +e
    awk -v label="$label" '
        /GPU\[0\].*VRAM Total Used Memory/ && $NF + 0 > peak {
            peak = $NF + 0
        }
        END {
            printf "vram label=%s peak_bytes=%.0f peak_gib=%.3f limit_gib=15.000\n",
                label, peak, peak / 1073741824
            if (peak <= 0 || peak > 15 * 1073741824) {
                exit 1
            }
        }
    ' "$log_dir/${label}-vram.log" | tee "$log_dir/${label}-vram.result"
    vram_status=${PIPESTATUS[0]}
    set -e
    if ((vram_status != 0)); then
        status=$vram_status
    fi
    return "$status"
}

run_arm vulkan vulkan
run_arm rocm-rocwmma rocm-rocwmma

for log in "$log_dir/vulkan.log" "$log_dir/rocm-rocwmma.log"; do
    grep -Eq '^precision .*loss_scale=1024([.]0)? .*nonfinite_count=0 ' "$log"
done

reference_tflops=$(sed -n \
    's/^result .*throughput=\([0-9][0-9.]*\) TFLOP\\/s.*/\1/p' \
    "$log_dir/vulkan.log" | tail -n 1)
candidate_tflops=$(sed -n \
    's/^result .*throughput=\([0-9][0-9.]*\) TFLOP\\/s.*/\1/p' \
    "$log_dir/rocm-rocwmma.log" | tail -n 1)
test -n "$reference_tflops"
test -n "$candidate_tflops"

awk -v reference="$reference_tflops" -v candidate="$candidate_tflops" 'BEGIN {
    change = 100 * (candidate / reference - 1)
    printf "performance reference=%.2f candidate=%.2f change=%+.2f%%\n",
        reference, candidate, change
}' | tee "$log_dir/performance.result"
