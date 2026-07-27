#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/roofline}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

sizes=${ROOFLINE_SIZES:-"4096 8192"}
dtypes=${ROOFLINE_DTYPES:-"f32 f16 bf16"}
successful_f32=0

for dtype in $dtypes; do
    for size in $sizes; do
        label="${dtype}-${size}"
        set +e
        cargo run --release --no-default-features --features vulkan \
            --example roofline -- \
            --size "$size" --dtype "$dtype" --warmup 1 --steps 3 \
            --max-steps 9 2>&1 | tee "$log_dir/${label}.log"
        status=${PIPESTATUS[0]}
        set -e
        printf 'dtype=%s size=%s status=%s\n' "$dtype" "$size" "$status" \
            | tee "$log_dir/${label}.status"
        if [[ $dtype == f32 && $status -eq 0 ]]; then
            successful_f32=$((successful_f32 + 1))
        fi
    done
done

# Every dtype/shape combination is deliberately a probe: a device-lost result
# at one size is itself roofline data. At least one fp32 reference must complete,
# however, or the harness/backend is too broken to interpret reduced precision.
if [[ $successful_f32 -eq 0 ]]; then
    echo "no fp32 roofline shape completed successfully" >&2
    exit 1
fi
