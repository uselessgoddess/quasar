#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/cross-entropy}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

common=(
    --model tiny-turbo
    --warmup 1
    # P0 exposed a late autotune in the first measured sample. Use the full
    # extended window in each arm so its median is insensitive to that sample.
    --steps 9
    --max-steps 9
    --dtype f32
    --ssd serial
    --checkpointing false
    --muon true
)

run_arm() {
    local label=$1
    local chunked=$2
    local micro_batch=$3
    local accum=$4
    local sampler_pid=""
    local status=0
    local vram_status=0

    if command -v rocm-smi >/dev/null 2>&1; then
        (
            while true; do
                date -u +'%Y-%m-%dT%H:%M:%SZ'
                rocm-smi --showmeminfo vram
                sleep 1
            done
        ) >"$log_dir/${label}-vram.log" 2>&1 &
        sampler_pid=$!
    fi

    set +e
    QUASAR_CHUNKED_CROSS_ENTROPY=$chunked \
        cargo run --release --no-default-features --features vulkan \
        --example train_bench -- "${common[@]}" \
        --micro-batch "$micro_batch" --accum "$accum" \
        2>&1 | tee "$log_dir/${label}.log"
    status=${PIPESTATUS[0]}
    set -e

    if [[ -n $sampler_pid ]]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
        awk -v label="$label" '
            /GPU\[0\].*VRAM Total Used Memory/ && $NF + 0 > peak {
                peak = $NF + 0
            }
            END {
                printf "vram label=%s peak_bytes=%.0f peak_gib=%.3f\n",
                    label, peak, peak / 1073741824
                if (peak <= 0 || peak > 15 * 1073741824) {
                    exit 1
                }
            }
        ' "$log_dir/${label}-vram.log" | tee "$log_dir/${label}-vram.result"
        vram_status=${PIPESTATUS[0]}
        if ((vram_status != 0)); then
            status=$vram_status
        fi
    fi
    return "$status"
}

# Reference and direct candidate isolate only the loss implementation. The
# batch-8 candidate then spends the released memory on a larger micro-batch;
# every arm still processes 131,072 identical tokens per optimizer step.
run_arm materialized-4x32 0 4 32
run_arm chunked-4x32 1 4 32
run_arm chunked-8x16 1 8 16

awk '
    function parsed_loss(line, fields, i) {
        split(line, fields, " ")
        for (i in fields) {
            if (fields[i] ~ /^loss=/) {
                sub(/^loss=/, "", fields[i])
                return fields[i] + 0
            }
        }
        return -1
    }
    FILENAME == ARGV[1] && /^(warmup|measured) / {
        reference[++nr] = parsed_loss($0)
        next
    }
    FILENAME == ARGV[2] && /^(warmup|measured) / {
        direct[++nd] = parsed_loss($0)
        next
    }
    FILENAME == ARGV[3] && /^(warmup|measured) / {
        large[++nl] = parsed_loss($0)
    }
    END {
        if (nr != 10 || nd != nr || nl != nr) {
            printf "quality gate expected 10 losses per arm, got %d/%d/%d\n",
                nr, nd, nl > "/dev/stderr"
            exit 1
        }
        worst = 0
        for (i = 1; i <= nr; i++) {
            first = i > 2 ? i - 2 : 1
            sum_r = sum_d = sum_l = count = 0
            for (j = first; j <= i; j++) {
                sum_r += reference[j]
                sum_d += direct[j]
                sum_l += large[j]
                count++
            }
            smooth_r = sum_r / count
            smooth_d = sum_d / count
            smooth_l = sum_l / count
            delta_d = (smooth_d - smooth_r) / smooth_r
            delta_l = (smooth_l - smooth_r) / smooth_r
            abs_d = delta_d < 0 ? -delta_d : delta_d
            abs_l = delta_l < 0 ? -delta_l : delta_l
            if (abs_d > worst) worst = abs_d
            if (abs_l > worst) worst = abs_l
            printf "quality step=%d reference=%.6f chunked4=%.6f (%+.4f%%) chunked8=%.6f (%+.4f%%)\n",
                i, smooth_r, smooth_d, delta_d * 100, smooth_l, delta_l * 100
        }
        result = worst > 0.005 ? "fail" : "pass"
        printf "quality result=%s max_smoothed_loss_delta=%.4f%% limit=0.5000%%\n",
            result, worst * 100
        if (worst > 0.005) exit 1
    }
' "$log_dir/materialized-4x32.log" "$log_dir/chunked-4x32.log" \
    "$log_dir/chunked-8x16.log" | tee "$log_dir/quality.result"

reference_throughput=$(sed -n \
    's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/materialized-4x32.log" | tail -n 1)
direct_throughput=$(sed -n \
    's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/chunked-4x32.log" | tail -n 1)
large_throughput=$(sed -n \
    's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/chunked-8x16.log" | tail -n 1)
test -n "$reference_throughput"
test -n "$direct_throughput"
test -n "$large_throughput"

awk -v reference="$reference_throughput" -v direct="$direct_throughput" \
    -v large="$large_throughput" 'BEGIN {
    direct_change = 100 * (direct / reference - 1)
    large_change = 100 * (large / reference - 1)
    printf "paired A/B: materialized4=%d chunked4=%d (%+.1f%%) chunked8=%d (%+.1f%%) tok/s\n",
        reference, direct, direct_change, large, large_change
    if (large < reference * 0.95) {
        print "chunked batch-8 candidate regressed by more than 5%" > "/dev/stderr"
        exit 1
    }
}' | tee "$log_dir/performance.result"
