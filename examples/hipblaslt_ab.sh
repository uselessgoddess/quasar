#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/hipblaslt}
mkdir -p "$repo_root/$log_dir" "$repo_root/target"
cd "$repo_root"

if [[ ! -x /opt/rocm/bin/hipcc ]]; then
    echo "/opt/rocm/bin/hipcc is required" >&2
    exit 1
fi
if [[ ! -f /opt/rocm/include/hipblaslt/hipblaslt.h ]]; then
    echo "hipBLASLt headers are required" >&2
    exit 1
fi

/opt/rocm/bin/hipcc -O3 -std=c++17 -Wall -Wextra -Werror \
    -I/opt/rocm/include -L/opt/rocm/lib \
    examples/hipblaslt_backend_spike.cpp -lhipblaslt \
    -o target/hipblaslt_backend_spike

sample_vram() {
    local label=$1
    while true; do
        date -u +'%Y-%m-%dT%H:%M:%SZ'
        rocm-smi --showmeminfo vram
        sleep 1
    done >"$log_dir/${label}-vram.log" 2>&1
}

run_arm() {
    local label=$1
    shift
    local sampler_pid=""
    local status=0
    local vram_status=0

    sample_vram "$label" &
    sampler_pid=$!
    set +e
    "$@" 2>&1 | tee "$log_dir/${label}.log"
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
            if (peak <= 0 || peak > 15 * 1073741824) exit 1
        }
    ' "$log_dir/${label}-vram.log" | tee "$log_dir/${label}-vram.result"
    vram_status=${PIPESTATUS[0]}
    set -e
    if ((vram_status != 0)); then
        status=$vram_status
    fi
    return "$status"
}

run_arm vulkan \
    cargo run --release --no-default-features --features vulkan \
    --example linear_backend_spike -- \
    --rows 4096 --input-features 640 --output-features 32768 \
    --warmup 1 --steps 3 --max-steps 9
run_arm hipblaslt target/hipblaslt_backend_spike

for log in "$log_dir/vulkan.log" "$log_dir/hipblaslt.log"; do
    grep -Eq '^precision .*loss_scale=1024([.]0+)? .*nonfinite_count=0 ' "$log"
done

reference_tflops=$(
    awk '/^result / {
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^throughput=/) {
                sub(/^throughput=/, "", $i)
                value = $i
            }
        }
    } END { print value }' "$log_dir/vulkan.log"
)
candidate_tflops=$(
    awk '/^result / {
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^throughput=/) {
                sub(/^throughput=/, "", $i)
                value = $i
            }
        }
    } END { print value }' "$log_dir/hipblaslt.log"
)
test -n "$reference_tflops"
test -n "$candidate_tflops"

awk -v reference="$reference_tflops" -v candidate="$candidate_tflops" 'BEGIN {
    change = 100 * (candidate / reference - 1)
    printf "performance reference=%.2f candidate=%.2f change=%+.2f%% minimum=+2.00%%\n",
        reference, candidate, change
    if (candidate < reference * 1.02) {
        print "hipBLASLt did not clear the paired throughput gate" > "/dev/stderr"
        exit 1
    }
}' | tee "$log_dir/performance.result"
