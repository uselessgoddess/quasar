#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/hipblaslt-tied-head}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

# Both arms use the complete accepted P5 recipe. The only variable is the
# backend: CubeCL Vulkan for the reference and CubeCL ROCm plus the native
# hipBLASLt tied output head for the candidate.
common=(
    --model tiny-turbo
    --micro-batch 4
    --accum 32
    --dtype f32
    --ssd serial
    --checkpointing false
    --muon true
    --head-dtype f16
    --ffn-dtype f16
    --mamba-dtype f16
    --precision-diagnostics true
)

run_arm() {
    local label=$1
    local phase=$2
    local feature=$3
    local sampler_pid=""
    local status=0
    local vram_status=0
    local phase_args=()

    case "$phase" in
        performance)
            phase_args=(--warmup 1 --steps 3 --max-steps 9 --vary-tokens false)
            ;;
        quality)
            # Match the accepted mixed-precision gates: twenty measured steps,
            # varying tokens, and the same seed in both fresh processes.
            phase_args=(--warmup 1 --steps 20 --max-steps 20 --vary-tokens true)
            ;;
        *)
            echo "unknown benchmark phase: $phase" >&2
            return 2
            ;;
    esac

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
        --example train_bench -- "${common[@]}" "${phase_args[@]}" \
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

run_arm performance-vulkan performance vulkan
run_arm performance-hipblaslt performance hipblaslt

reference_throughput=$(sed -n \
    's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/performance-vulkan.log" | tail -n 1)
candidate_throughput=$(sed -n \
    's/^result .*throughput=\([0-9][0-9]*\) tok\/s.*/\1/p' \
    "$log_dir/performance-hipblaslt.log" | tail -n 1)
test -n "$reference_throughput"
test -n "$candidate_throughput"

awk -v reference="$reference_throughput" -v candidate="$candidate_throughput" 'BEGIN {
    change = 100 * (candidate / reference - 1)
    printf "performance reference=%d candidate=%d change=%+.2f%% minimum=+2.00%%\n",
        reference, candidate, change
    if (candidate < reference * 1.02) {
        print "hipBLASLt tied head did not clear the paired throughput gate" > "/dev/stderr"
        exit 1
    }
}' | tee "$log_dir/performance.result"

run_arm quality-vulkan quality vulkan
run_arm quality-hipblaslt quality hipblaslt

for log in \
    "$log_dir/performance-vulkan.log" \
    "$log_dir/performance-hipblaslt.log" \
    "$log_dir/quality-vulkan.log" \
    "$log_dir/quality-hipblaslt.log"
do
    grep -Eq '^precision .*loss_scale=1024([.]0)? .*nonfinite_count=0 ' "$log"
    awk '
        /^(warmup|measured) / {
            points++
            found = 0
            for (i = 1; i <= NF; i++) {
                if ($i == "loss_scale=1024") found = 1
            }
            if (!found) {
                print "unexpected loss scale: " $0 > "/dev/stderr"
                bad = 1
            }
        }
        END {
            if (points == 0 || bad) exit 1
        }
    ' "$log"
done

# Compare every prefix of the paired trajectory using the accepted
# trailing-three smoother and ±0.5% loss threshold.
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
        if (nr != nc || nr != 21) {
            printf "quality gate expected 21 paired losses, got %d/%d\n", nr, nc > "/dev/stderr"
            exit 1
        }
        worst = 0
        for (i = 1; i <= nr; i++) {
            first = i > 2 ? i - 2 : 1
            sum_r = sum_c = count = 0
            for (j = first; j <= i; j++) {
                sum_r += reference[j]
                sum_c += candidate[j]
                count++
            }
            smooth_r = sum_r / count
            smooth_c = sum_c / count
            relative = smooth_r == 0 ? (smooth_c == 0 ? 0 : 1) : (smooth_c - smooth_r) / smooth_r
            absolute = relative < 0 ? -relative : relative
            if (absolute > worst) worst = absolute
            printf "quality step=%d vulkan=%.6f hipblaslt=%.6f relative=%+.4f%%\n",
                i, smooth_r, smooth_c, relative * 100
        }
        result = worst > 0.005 ? "fail" : "pass"
        printf "quality result=%s max_smoothed_loss_delta=%.4f%% limit=0.5000%%\n",
            result, worst * 100
        if (worst > 0.005) exit 1
    }
' "$log_dir/quality-vulkan.log" "$log_dir/quality-hipblaslt.log" \
    | tee "$log_dir/quality.result"
