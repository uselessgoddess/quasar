# quasar-factorio

Everything between "no corpus" and "look at what it built": a Factorio blueprint
corpus, the grammar quasar reads it in, a grader that scores generations the way
the game would, and a renderer so the output can be looked at rather than
squinted at.

The model itself is the `nano` preset in the Rust crate one directory up. This
package is the harness around it, and it is deliberately dependency-free —
`pyproject.toml` declares no runtime dependencies at all. PNG comes out of
`zlib` and `struct`, the plots are drawn into the same framebuffer as the
blueprints, and CI builds a corpus without downloading a wheel.

## The model

`quasar-nano` is 3.5M parameters, sized for this corpus rather than scaled down
from `tiny`:

```
$ cargo run --release -- budget nano
embedding       0.1M      seq_len          512
lm_head         0.0M      ssd chunk         32
ssm             1.5M      fwd FLOPs/token  8.1M
attention       0.2M      states muon      0.04 GiB
ffn             1.8M      activations      120 MiB at micro_batch 1
total           3.5M      micro_batch in 16 GiB, muon states 135
```

Three numbers decide the shape. The vocabulary is **495 tokens** instead of
32,768, so the embedding costs almost nothing and the whole budget goes into the
stack. The longest document in the corpus is **460 tokens**, so `seq_len 512`
holds a whole blueprint and there is nothing past it worth attending to. And
attention is **unwindowed**, which is the one place `nano` disagrees with every
larger preset: the task is placing entities that must not overlap ones already
placed, a 64-entity blueprint is 400 tokens of history that all of it depends
on, and a 128-token window would hide two thirds of the design being built.
Quadratic attention over 512 tokens at `d_model 192` is a rounding error against
the SSD scan.

At `micro_batch 32` the run uses a small fraction of the 135 micro-batches that
fit 16 GiB, so it cannot run out of memory on the card it was written for.

## The grammar

One blueprint is one line. A `<spec>` states the intent, and everything after it
is the design that satisfies it:

```
<bp> <spec> k:mining-outpost r:none #30 #7 #18 </spec>
     <e> electric-mining-drill x00 y00 d2
     <e> transport-belt x03 y00 d0
     <e> electric-mining-drill x04 y00 d6 ... </bp>
```

`k:` is the kind, `r:` the product, and the three counts are entities, width and
height. Coordinates are zero-padded on a 64×64 grid, `d0`–`d7` is the eight-way
direction the game uses, and assemblers carry `r:RECIPE` while inserters and
belts can carry `t:input` or `t:output`. Cutting the line after `</spec>` gives
a prompt: the model is asked to build something to a specification it has never
been trained on.

Every name in the vocabulary is a real prototype. `assets/prototypes.json` is
60 entities, 153 items and 212 recipes distilled from Factorio's own `data.raw`
by `tools/distill_data_raw.py`, which records the source URL, the game version
and the sha256 of the bytes it read — so entity sizes, recipe ingredients and
crafting times are traceable to the game rather than to somebody's memory of it.

## Pipeline

All of it, end to end, is one script:

```sh
BACKEND=vulkan examples/factorio.sh runs/nano
```

That is what CI runs on the maintainer's card, and it is sized to finish inside
half an hour: it builds the corpus, previews it, trains, generates from *every*
checkpoint the run wrote, grades the newest, draws the board and leaves a
blueprint string behind. `STEPS`, `DESIGNS` and `PROMPTS` are the knobs;
`PROMPTS` is the one that decides the wall clock, because sampling is the
expensive half.

Underneath it is a sequence of commands that each stand on their own:

```sh
# a corpus quasar train can read: shards, tokenizer, held-out prompts
python -m quasar_factorio.cli build corpus --count 20000

# what went in, before spending a GPU on it
python -m quasar_factorio.cli preview corpus/preview.png --corpus corpus --count 12
python -m quasar_factorio.cli heatmap corpus/occupancy.png --count 400

# train, from the repository root
cargo run --release --no-default-features --features vulkan -- \
    train nano --data corpus --out runs/nano 2>&1 | tee runs/nano/train.log

# generate against the specs it was never trained on. Naming a checkpoint
# rather than the run samples the model as it was at that step, which is what
# the grade-over-training panel and the timelapse frames are made of.
cargo run --release --features vulkan -- generate runs/nano/step_0000400 \
    --tokenizer corpus/tokenizer.json \
    --prompts corpus/prompts.jsonl --out runs/nano/samples-000400.jsonl --tokens 460

# score it, draw it, plot it
python -m quasar_factorio.cli grade runs/nano/samples.jsonl \
    --sheet runs/nano/sheet.png --json runs/nano/grade.json
python -m quasar_factorio.cli plot runs/nano/train.log runs/nano/metrics.png \
    --grade runs/nano/samples-*.jsonl

# and the end of the whole thing: something to paste into the game
python -m quasar_factorio.cli export design.txt | xclip -selection clipboard
```

Installed as a script, the commands are `quasar-factorio build`, `… grade` and
so on; run out of the source tree, `PYTHONPATH=src python -m quasar_factorio.cli`
does the same without installing anything.

Every command reads something off disk, writes something to disk and prints what
it did. That is what makes the loop debuggable: every intermediate is sitting
there to be looked at, and any step can be re-run on its own without rebuilding
the ones before it.

## What the corpus is

Ten generators — smelter columns, assembler rows, mall cells, bus taps,
balancers, mining outposts, solar and oil blocks, lab blocks, belt lanes — draw
layouts with real entity footprints, real recipes and real ingredient ratios.
Each layout is then augmented into its symmetries (four rotations, two
reflections, belt tiers), which is where most of the documents come from.

Two decisions matter more than the rest:

*Nothing invalid gets in.* Every document is graded before it is written, and a
design the grader rejects is dropped rather than repaired. The generators are
held to a perfect score by their own tests, so `rejected` in `manifest.json` is
an alarm, not a tolerance.

*The split is by design, not by document.* Holding out every Nth document would
put a smelter column in `train` and its mirror image in `valid`, and validation
loss would then be measuring memorisation. Whole designs move together,
augmentations and all, and "the same design" is decided by a canonical form that
sees through rotation, reflection and belt tier.

A 20,000-draw build measures 4,408 distinct layouts, 38,405 documents and 7.46M
training tokens at a 495-token vocabulary. The other 15,592 draws were forms of
a layout already kept, and 41,595 of the expanded documents came out
byte-identical to one already written — drawing at random from ten generators
collides, and the manifest says by how much. The 6,000 draws
`examples/factorio.sh` defaults to give 2,439 layouts and 3.17M tokens, already
more than a 3.5M-parameter model gets through in half an hour.

The corpus is entirely synthetic, and nothing here fetches from the network:
what a run trains on has to be reproducible from a seed and a pinned
`prototypes.json`. Human blueprints — factorioprints.com and the like — are
still the obvious source of the variety a generator does not invent, so
`dataset.build` takes an `extra` iterator of `Design`s that is deduplicated,
graded and split by exactly the rules the drawn ones are. Scraping into it is a
script that does not exist yet, not a change to the pipeline.

## How output is judged

Loss says the model is fitting the corpus. Only the grader says it is learning
to build something the game would accept, so `grade` reports both the hard
constraints and the soft ones:

| | |
| --- | --- |
| `parses` | it is a document at all |
| `valid` | no overlaps, on the grid, no illegal recipes |
| `spec honoured` | the entity count, width and height it was asked for |
| `powered` | machines within reach of a pole |
| `inserters connected` | an inserter with something on both sides |
| `belts lead somewhere` | a belt whose next tile is not empty ground |
| `quality` | zero if invalid, else the mean of the three above |

Failures are counted by reason, so a run that scores zero still says *why* — and
the contact sheet is sorted by score rather than sampled, because the
interesting frames of a training run are the extremes and a random twelve from
five hundred shows neither.

`--order given` is the exception. It keeps the sheet in prompt order, which
makes one sheet per checkpoint into a timelapse: the same specification stays in
the same cell while the model gets better at satisfying it. `examples/factorio.sh`
writes those into `frames/`, and any of

```sh
ffmpeg -framerate 4 -pattern_type glob -i 'runs/nano/frames/*.png' training.mp4
```

turns them into the video.

## Looking at it

The renderer has no textures on purpose. Entity footprints are exact, colours
come from the prototypes, a bar on the leading edge says which way a belt or
inserter faces, and a recipe shows as a pip in the middle of the machine. That
fits a whole design into a card small enough to tile forty of them onto one
sheet, which a texture atlas does not, and it renders in milliseconds with
nothing installed — which is what makes per-checkpoint frames and timelapses
cheap.

`plot` draws the same way: near-black ground, hairline grid, hard edges, the
same 5×7 bitmap font, and every number that matters printed as text rather than
left to be read off an axis. Its input is whatever `quasar train` wrote to
stdout, because the log always exists — it exists for runs that crashed, and for
runs that happened before the plotting code did.

## Development

```sh
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

`uv` installs a linter and a test runner and nothing else — the package itself
still has no runtime dependencies, and `[dependency-groups]` in `pyproject.toml`
is the whole of it. CI runs exactly those three commands.

The suite is fast and has no GPU in it. `experiments/saturation.py` measures how
many distinct designs the generators can actually produce before they start
repeating themselves, which is the number that decides whether the corpus is
worth enlarging.
