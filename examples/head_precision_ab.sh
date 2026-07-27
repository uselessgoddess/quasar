#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/head-precision}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

common=(
    --model tiny-turbo
    --micro-batch 4
    --accum 32
    --warmup 1
    # P0 found a second autotune in the first measured sample. Nine samples for
    # both arms keep the paired comparison independent of that outlier.
    --steps 9
    --max-steps 9
    --dtype f32
    --ssd serial
    --checkpointing false
    --muon true
    --precision-diagnostics true
)

run_arm() {
    local label=$1
    shift
    local sampler_pid=""
    local status=0
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
    cargo run --release --no-default-features --features vulkan \
        --example train_bench -- "${common[@]}" "$@" \
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
            }
        ' "$log_dir/${label}-vram.log" | tee "$log_dir/${label}-vram.result"
    fi
    return "$status"
}

run_arm fp32
run_arm bf16-head --head-dtype bf16

# Compare a trailing three-step moving average. The warm-up plus nine measured
# optimizer steps are deterministic (same seed and synthetic tokens) and long
# enough to expose immediate precision divergence without occupying the shared
# card for a 2K-step corpus run.
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
    FNR == NR && /^(warmup|measured) / {
        reference[++nr] = parsed_loss($0)
        next
    }
    FNR != NR && /^(warmup|measured) / {
        candidate[++nc] = parsed_loss($0)
    }
    END {
        if (nr != nc || nr != 10) {
            printf "quality gate expected 10 paired losses, got %d/%d\n", nr, nc > "/dev/stderr"
            exit 1
        }
        worst = 0
        for (i = 1; i <= nr; i++) {
            first = i > 2 ? i - 2 : 1
            sum_r = 0
            sum_c = 0
            count = 0
            for (j = first; j <= i; j++) {
                sum_r += reference[j]
                sum_c += candidate[j]
                count++
            }
            smooth_r = sum_r / count
            smooth_c = sum_c / count
            if (smooth_r == 0) {
                relative = smooth_c == 0 ? 0 : 1
            } else {
                relative = (smooth_c - smooth_r) / smooth_r
            }
            absolute = relative < 0 ? -relative : relative
            if (absolute > worst) {
                worst = absolute
            }
            printf "quality step=%d fp32=%.6f bf16_head=%.6f relative=%+.4f%%\n",
                i, smooth_r, smooth_c, relative * 100
        }
        result = worst > 0.005 ? "fail" : "pass"
        printf "quality result=%s max_smoothed_loss_delta=%.4f%% limit=0.5000%%\n",
            result, worst * 100
        if (worst > 0.005) {
            exit 1
        }
    }
' "$log_dir/fp32.log" "$log_dir/bf16-head.log" \
    | tee "$log_dir/quality.log"
