#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
log_dir=${1:-benchmark-logs/roofline}
mkdir -p "$repo_root/$log_dir"
cd "$repo_root"

sizes=${ROOFLINE_SIZES:-"4096 8192"}
dtypes=${ROOFLINE_DTYPES:-"f32 f16 bf16"}
failed_f32=0

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
        if [[ $dtype == f32 && $status -ne 0 ]]; then
            failed_f32=1
        fi
    done
done

# Reduced precision is deliberately a probe: unsupported bf16/f16 is a result
# to document, while an fp32 failure means the roofline harness itself is bad.
exit "$failed_f32"
