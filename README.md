# quasar

A Mamba-3 language model family trained end to end on one consumer GPU, in Rust
on [burn](https://github.com/tracel-ai/burn) + wgpu.

Three language presets plus a Factorio family, all hybrid stacks of Mamba-3
blocks with a GQA layer every fourth to seventh position — sliding-window
everywhere except `factorio-nano`, whose context is one whole blueprint and has
nothing to slide over:

| | params | fwd FLOPs/token | states fp32 | activations / micro-batch | micro-batches in 16 GiB |
| --- | --- | --- | --- | --- | --- |
| `factorio-nano` | 3.6M | 8.2M | 0.05 GiB | 0.12 GiB | 134 |
| `tiny-turbo` | 62.6M | 129.8M | 0.93 GiB | 1.09 GiB | 14 |
| `tiny` | 146.7M | 329.4M | 2.19 GiB | 6.62 GiB | 2 |
| `base` | 1117.5M | 2306.2M | 16.65 GiB | 24.48 GiB | 0 |

`docs/DESIGN.md` justifies every number above, states the training-time budget
honestly, and explains why this is not a mixture of experts.
[`docs/MEMORY.md`](docs/MEMORY.md) takes apart the last two columns — where the
VRAM actually goes, which burn-mamba setting moves it, and what `tiny-turbo`
gives up to fit fourteen estimated micro-batches where `tiny` fits two.

`tiny` and `tiny-turbo` share an 8192-token vocabulary; only `base` is large
enough for 32768 to be worth its embedding, its head GEMM and its logits.

## Pipeline

```sh
# what a preset costs before committing a week to it
cargo run --release -- budget tiny

# a corpus: parquet, jsonl or txt, files or directories
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
    --include "sample/10BT/*" --local-dir data/fineweb-edu

cargo run --release -- tokenizer data/fineweb-edu --vocab-size 8192
cargo run --release -- prepare data/fineweb-edu --out data/shards
cargo run --release -- train tiny --data data/shards --out runs/tiny

# measured RX 9070 XT recipe: 640×12, batch 4, serial SSD, no checkpoint replay
cargo run --release --no-default-features --features vulkan -- \
    train tiny-turbo --data data/shards --out runs/turbo

cargo run --release -- eval runs/tiny --data data/shards
cargo run --release -- generate runs/tiny --prompt "The reason"
```

`train` resumes from the newest checkpoint under `--out`, so a run interrupted
at any point continues where it stopped. Overrides worth knowing:
`--steps`, `--micro-batch`, `--accum`, `--lr`, `--warmup`, `--decay`,
`--save-every`, `--eval-every`, `--muon`, `--checkpointing`, `--ssd`.
`tiny-turbo` also defaults to
`--head-dtype f16 --ffn-dtype f16 --mamba-dtype f16`; pass any option as
`fp32` to disable that independently measured precision path.

The default tiny recipe is 12,500 optimizer steps, or 3.2768B tokens with the
default `8 × 16 × 2048` effective batch. Changing either batch knob also changes
the total token budget unless `--steps` is adjusted. The startup summary prints
both quantities before training begins. A progress item in the dashboard is one
optimizer step, not one sequence or token.

When training is attached to a terminal it opens Burn's official TUI, with
live plots for training/validation loss, perplexity, bits-per-byte, learning
rate, throughput, tokens processed, ETA, and effective TFLOP/s. Press `q`, then
`s`, to stop cleanly; the loop writes a resumable checkpoint before exiting.
Redirected output and CI keep the line-oriented logs instead. See
[`docs/TRAINING_SPEED.md`](docs/TRAINING_SPEED.md) for interpreting these
numbers and the investigation behind the defaults.

Three knobs decide the memory/speed tradeoff. Muon is on for every preset.
Checkpointing is on for the larger presets, while `tiny-turbo` uses the faster
measured combination `--micro-batch 4 --accum 32 --checkpointing false`:
`--muon false` puts the hidden matrices back on AdamW (16 B/param of state
instead of 12, which is what stopped `base` fitting 16 GB), and
`--checkpointing false` stops recomputing activations in the backward, trading
memory for speed. `tiny-turbo` also defaults to measured `--ssd serial`, which
retains burn-mamba's chunk intermediates. Together with one CubeCL stream and a
640×12 shape, the final matched experiment reached 10.37k tok/s on a 16-GB RX
9070 XT; the fp32 full production batch reached 10.55k tok/s. Casting only its
tied output-head GEMM to f16, with fp32 master/logits/loss and dynamic loss
scaling, raised the same batch to **12.13k tok/s** at 12.111 GiB peak VRAM while
the paired smoothed loss stayed within 0.1243%. Extending that measured path to
the three FFN projections raised it again to **14.68k tok/s** at 12.038 GiB;
the paired smoothed loss stayed within 0.0060%. Extending it once more to the
Mamba input/output projections reached **17.69k tok/s** at 11.835 GiB, with
the paired smoothed loss again within 0.0060%. SSD coefficients,
discretization, recurrent state, norms, elementwise operations, residuals and
master weights remain fp32. The vendored burn-mamba branch uses a measured
fused CubeCL rank-one scan by default and
retains `BURN_MAMBA_FUSED_SINGLE_SCAN=0` as a reference-path escape hatch. Its
backward replays each eight-token block forward from a checkpoint; dividing the
decay back out instead is cheaper, and cost `tiny-turbo` its gradients around
step 70 at every precision (issue #23).
Select `--checkpointing true --ssd recalculated` if a larger override runs out
of memory. Other presets retain the memory-saving defaults. See
`docs/DESIGN.md` §3, [`docs/KERNELS.md`](docs/KERNELS.md), and the
[`docs/ROOFLINE.md`](docs/ROOFLINE.md) precision gate.

Validation reports negative log-likelihood, perplexity and **bits-per-byte** —
the last is the only figure comparable across tokenizers, and the one the design
targets are written in.

The first GPU step is not representative of training speed or peak live tensor
memory. With the GPU features, Burn compiles fused kernels and benchmarks
candidate implementations for the shapes it sees; utilization therefore comes
in bursts while VRAM grows before the steady loop begins. `budget` now prints an
`activations` breakdown and the largest micro-batch that fits 16 GiB alongside
the `states` figures, so the answer is available before a run allocates anything
— but it is an analytic estimate of the tensors autodiff must retain, not a
measurement, and it does not cover fusion/autotuning workspaces or the backend
allocator's cache. `--micro-batch`, `--seq-len`, `--state-rank`, `--mimo-rank`,
`--expand`, `--attn-window`, `--attn-period` and `--ssd-chunk` all work on
`budget`, so a shape can be swept without rebuilding a preset. Still start at
`--micro-batch 1` on new hardware and raise it once the first optimizer step has
completed; change `--accum` inversely if the effective token batch must remain
fixed.

## Backends

The default is a CPU backend, so `cargo test` needs no GPU. Training wants one:

```sh
# RDNA4 and anything else with a Vulkan driver
cargo run --release --no-default-features --features vulkan -- train tiny

# the same card through ROCm/HIP, which needs the ROCm toolchain installed
cargo run --release --no-default-features --features rocm -- train tiny
```

Available: `flex` (CPU, default), `ndarray` (CPU), `wgpu`, `vulkan`, `rocm`,
`cuda`. On AMD, `vulkan` is the same runtime as `wgpu` compiled to SPIR-V rather
than WGSL, which the drivers handle better; `rocm` goes through HIP instead, and
which of the two is faster on RDNA4 is a question for a measurement, not for a
README. It needs a ROCm installation whose `hipcc` knows the card's target
(`gfx1201` for RX 9070 XT); `rocminfo` says what it is.

All four GPU features go through `gpu`, which turns on fusion in burn *and* in
burn-mamba together — burn's GPU backends are `Fusion<CubeBackend<_>>`, and
burn-mamba only implements its SSD extension for `Fusion` when its own `fusion`
feature is on.

The normal `release` profile deliberately skips LTO so GPU experiments do not
pay a full-program link on every iteration. For an infrequent final local build,
opt in with `cargo build --profile release-lto`; that separate profile enables
thin LTO and a single codegen unit.

## Trying it without a GPU

```sh
examples/smoke.sh
```

Fits a tokenizer, shards a synthetic corpus, trains 50 steps, evaluates and
samples — the whole pipeline in under a minute on a CPU.

## Checking that a recipe trains

```sh
BACKEND=vulkan examples/stability.sh
```

The recipe of issue #23 — `tiny-turbo` at micro-batch 4 — past the steps it
reported coming apart at, in fp32, fp16 on the head, fp16 everywhere and bf16
everywhere. Each arm has to finish with a loss below the uniform baseline, so an
arm that survives by learning nothing fails too. A unit test can show the
trainer rejects a non-finite gradient it is handed; only the card can show
whether a recipe produces one, which is why this runs on the self-hosted runner
as the `stability` job.

There is one card behind that runner, so `stability`, `gpu-benchmark` and
`backend-spike` do not run per commit: they run on push to `main`, and otherwise
only when somebody asks for the number by name —
`gh workflow run ci.yml --ref <branch> -f jobs=stability`, or, from a branch in
a fork, `[ci: stability]` in the pull request description, which the next push
picks up. Both selectors take `all`, `stability`, `benchmark` or `blueprints`,
so asking about convergence does not also spend an hour on throughput. Take the
line back out of the description once the answer is in.

## Factorio blueprints

[`factorio/`](factorio/README.md) is a harness that trains `factorio-nano` to build
Factorio blueprints instead of text: a synthetic corpus with the game's real
sizes and recipes — plus, once the generators run out of things to say, human
blueprints fetched from factorioprints and held to the same grader — a grammar a
blueprint round-trips through, a grader that says whether a generated design
would power up, and a metrics board. The whole run —
corpus, training, sampling from every checkpoint, grading, plots — is one
script, and it takes nine minutes on the same card the language presets are
sized for.

```sh
BACKEND=vulkan examples/factorio.sh runs/nano
```

## Development

```sh
cargo fmt --package quasar --check
cargo clippy --all-targets -- -D warnings
cargo test
```
