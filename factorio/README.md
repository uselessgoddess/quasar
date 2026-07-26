# quasar-factorio

Everything between "no corpus" and "look at what it built": a Factorio blueprint
corpus, the grammar quasar reads it in, a grader that scores generations the way
the game would, and a renderer so the output can be looked at rather than
squinted at.

The model itself is the `factorio-nano` preset in the Rust crate one directory
up — its own model family in `src/config/factorio.rs`, sized by this corpus
rather than by a parameter class, with room for a larger member once the grid
grows. This
package is the harness around it, and it is deliberately dependency-free —
`pyproject.toml` declares no runtime dependencies at all. PNG comes out of
`zlib` and `struct`, the plots are drawn into the same framebuffer as the
blueprints, and CI builds a corpus without downloading a wheel.

## The model

`quasar-factorio-nano` is 3.5M parameters, sized for this corpus rather than scaled down
from `tiny`:

```
$ cargo run --release -- budget factorio-nano
embedding       0.1M      seq_len          512
lm_head         0.0M      ssd chunk         32
ssm             1.5M      fwd FLOPs/token  8.1M
attention       0.2M      states muon      0.04 GiB
ffn             1.8M      activations      120 MiB at micro_batch 1
total           3.5M      micro_batch in 16 GiB, muon states 135
```

Three numbers decide the shape. The vocabulary is **739 tokens** instead of
32,768, so the embedding costs almost nothing and the whole budget goes into the
stack. The longest document in the corpus is **460 tokens**, so `seq_len 512`
holds a whole blueprint and there is nothing past it worth attending to. And
attention is **unwindowed**, which is the one place this family disagrees with every
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

The spec has two optional sections, used by the module task and absent from
everything the older generators write — a zone with declared inputs and outputs,
and the recipe chain that fills it:

```
<bp> <spec> k:module r:electronic-circuit #5 #16 #20
     <in> i:copper-plate s:n x03 y00
     <in> i:iron-plate s:n x00 y06
     <out> i:electronic-circuit s:e x15 y12
     <plan> assembling-machine-1 r:copper-cable #2
            assembling-machine-1 r:electronic-circuit #3 </plan>
     </spec> <e> ... </bp>
```

A port is the tile it feeds *inside* the design plus the edge it sits on, rather
than an offset along that edge: the tile rotates exactly like an entity, so
augmentation cannot desynchronise it from the design, and the model already
knows what `x03 y00` means where an offset would be a third numbering scheme
with the same shape as the other two. Being on the edge is what makes a module
composable — two modules whose ports agree butt together and the belts line up,
which a design with its inputs somewhere in the middle cannot promise.

Every name in the vocabulary is a real prototype. `assets/prototypes.json` is
60 entities, 235 items, 9 fluids and 212 recipes distilled from Factorio's own
`data.raw` by `tools/distill_data_raw.py`, which records the source URL, the
game version and the sha256 of the bytes it read — so entity sizes, recipe
ingredients and crafting times are traceable to the game rather than to
somebody's memory of it.

The item table is swept out of every prototype category rather than out of
`data.raw.item`, because Factorio files a science pack under `tool`, a piercing
round under `ammo` and a speed module under `module`. Reading only `item` missed
61 of the 214 things vanilla recipes name, and since the harness reads "has a
stack size" as "a belt can carry it", the missing rows did not read as missing
data — they read as `automation-science-pack` being a fluid, which deleted the
whole science branch from the planner's catalogue.

## Pipeline

All of it, end to end, is one script:

```sh
BACKEND=vulkan examples/factorio.sh runs/nano
```

That is what CI runs on the maintainer's card, and it is sized to finish inside
an hour: it builds the corpus, previews it, trains for the Chinchilla length the
preset carries, generates from *every* checkpoint the run wrote, grades the
newest, draws the board and leaves a blueprint string behind. `STEPS`, `DESIGNS`
and `PROMPTS` are the knobs; `PROMPTS` is the one that decides the wall clock,
because sampling is the expensive half. `REAL` points at a cache of human
blueprints and is used if the file is there — no cache, synthetic corpus, same
run otherwise.

Underneath it is a sequence of commands that each stand on their own:

```sh
# a corpus quasar train can read: shards, tokenizer, held-out prompts
python -m quasar_factorio.cli build corpus --count 20000

# what went in, before spending a GPU on it
python -m quasar_factorio.cli preview corpus/preview.png --corpus corpus --count 12
python -m quasar_factorio.cli heatmap corpus/occupancy.png --count 400

# train, from the repository root
cargo run --release --no-default-features --features vulkan -- \
    train factorio-nano --data corpus --out runs/nano 2>&1 | tee runs/nano/train.log

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

Eleven generators — smelter columns, assembler rows, mall cells, bus taps,
balancers, mining outposts, solar and oil blocks, lab blocks, belt lanes and
modules — draw layouts with real entity footprints, real recipes and real
ingredient ratios. Each layout is then augmented into its symmetries (four
rotations, two reflections, belt tiers), which is where most of the documents
come from.

The eleventh is the one the rest of this harness is now aimed at, and it is not
drawn the way the others are. `plan.solve` expands "electronic circuits, given
iron and copper plate" into a chain of recipes and whole numbers of machines by
reading `assets/prototypes.json` — the same ingredient lists, craft times and
machine speeds the game uses — and `synth.module` places what the planner
counted. The split is a compiler's: the planner is the front end and decides
*what* to build, the model is the back end and decides *where it goes*. The plan
rides in the prompt, so the model is conditioned on the ratios rather than asked
to invent them; a mis-remembered ratio produces a factory that looks perfect and
starves, and there is no reason to buy a probabilistic version of a table that
is already exact.

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

A 20,000-draw build measures 5,415 distinct layouts, 46,340 documents and 9.02M
training tokens at a 657-token vocabulary. The other 14,585 draws were forms of
a layout already kept, and 33,660 of the expanded documents came out
byte-identical to one already written — drawing at random from eleven generators
collides, and the manifest says by how much. The 6,000 draws
`examples/factorio.sh` defaults to give 2,729 layouts and 3.46M tokens, already
more than a 3.5M-parameter model gets through in half an hour.

Building it is pure Python and runs before the GPU gets to do anything, so what
it costs is worth knowing: `experiments/corpus_cost.py` prints the breakdown and
the answer is that the generators are not it. Almost all of the time goes into
deduplication — `augment.canonical` puts every design through eight rigid
motions to compare them, and each of those rebuilds every `Placement` three
times over. That is where `Placement.moved` and the running-minimum `bounds`
come from; they are unremarkable code with a measured reason.

### Where the generators run out

Raising `--count` past that stops helping, and `experiments/saturation.py` says
where. 3,200 draws from each of the eleven generators — 35,200 in all — yield
7,579 distinct layouts between them:

| generator | 200 draws | 800 | 3200 | distinct per draw |
| --- | ---: | ---: | ---: | ---: |
| belt-lane | 192 | 730 | 2684 | 84% |
| bus-tap | 183 | 592 | 1400 | 44% |
| module | 156 | 460 | 1235 | 39% |
| assembler-row | 181 | 540 | 1021 | 32% |
| mall-cell | 168 | 404 | 551 | 17% |
| solar-block | 125 | 210 | 216 | 7% |
| balancer | 109 | 176 | 184 | 6% |
| oil-block | 99 | 119 | 120 | 4% |
| smelter-column | 59 | 67 | 68 | 2% |
| lab-block | 53 | 56 | 56 | 2% |
| mining-outpost | 37 | 43 | 44 | 1% |

Six of the eleven are finished by their four-hundredth draw. A mining outpost
has 44 forms in total, a lab block 56 — every further draw is one of those
again. Two still climb at 3,200: the belt lane, which is the least interesting
thing in the corpus because its parameter space is large precisely to the extent
that nothing constrains it, and the module generator, whose space is the product
of a target item, a zone, and where its ports sit. So the synthetic ceiling is
roughly 7,600 layouts, about 12M tokens — the eleventh generator moved it by a
fifth, and it is the only one of the eleven that got there by adding a new thing
to say rather than a new way to say the same thing.

That is the number the training budget has to be read against. `factorio-nano`
is 3.5M parameters, so the Chinchilla rule asks for 70.4M training tokens — six
epochs over everything the generators can produce, and eight over what a
20,000-draw build contains. Data-constrained scaling laws put the point where
repeating stops being nearly free at about four epochs, which means a
Chinchilla-budget run needs ~17.6M *unique* tokens and the generators cannot get
there alone. More variety, not more draws.

### Human blueprints

So `dataset.build` takes an `extra` iterator of `Design`s, and `real.py` fills it
from blueprints people published. They are deduplicated, graded and split by
exactly the rules the drawn ones are — same `validate.grade`, same symmetries,
same canonical layout key — so this is an addition to the mixture rather than a
second pipeline.

```sh
python factorio/tools/fetch_blueprints.py --count 6000       # cache, resumable
python -m quasar_factorio.cli build corpus --count 20000 \
    --real factorio/data/blueprints.jsonl
```

There is no ready-made dataset to download. The Hugging Face hub has eight
Factorio datasets and not one of them is blueprint strings: the closest,
`piebro/factorio-blueprint-visualizations`, is under a thousand PNG renders for
text-to-image work (CC0), and the rest are wiki text, forum posts and save-file
metadata. So the source is `factorioprints.com`, which is a Firebase app with a
public read API — `blueprintSummaries.json?shallow=true` lists 17,780 ids and
`blueprints/<id>.json` returns one record, which is why the fetcher is a hundred
lines rather than a scraper. (`factorio.school` hosts a second collection; its
API answered 502 for the whole of this work, so it is named in the fetcher's
docstring rather than implemented.) Ids are Firebase push keys, so they sort by
upload date, and `--order spread` takes an even stride across the whole archive:
a prefix would be all 0.15, a suffix all Space Age.

The cache is not committed — it is other people's work and hundreds of megabytes
— so `data/` is gitignored and a manifest with the sha256 of the file is what a
run can cite.

Most of what is uploaded cannot be used, and the counts say which limit does the
throwing away. The whole archive is 17,477 records — 185 of them in the pre-0.15
string format this harness cannot read, all at the oldest end — and 84,611
blueprints once the books are walked, because a quarter of the records are books
and a book holds seventeen blueprints on average:

| | |
| --- | ---: |
| kept | 12,067 |
| too many entities (>64) | 28,491 |
| modded, or 2.0 entities | 25,316 |
| fewer than 4 entities | 8,774 |
| curved rails | 5,555 |
| no entities at all | 2,055 |
| graded invalid | 2,015 |
| larger than 64x64 | 338 |

14% survives, for 4,716 distinct layouts, 48,126 documents and 7.81M tokens —
mean 30 entities in a 10x9 footprint. That is what changes the arithmetic. The
generators top out around 11M tokens and the Chinchilla budget is 70.4M, so
synthetic-only is seven passes over everything they can say, where the
data-constrained scaling laws put the point at which repeating stops being
nearly free at four. 11M + 7.8M is 18.8M unique tokens against the 17.6M that
four epochs of the budget need: the human half is not a garnish here, it is what
moves the run from seven epochs to under four.

The single largest rejection class is size: what people
publish is whole factories, and the grammar addresses a 64x64 tile grid with at
most 64 entities. Cropping them to fit would be easy and wrong — `augment`
exists because a corpus containing broken output teaches that broken output is
sometimes correct, and a truncated blueprint is exactly that.

Two more rules are worth stating because they are the ones that would quietly
poison the corpus if relaxed. A blueprint is kept only if 90% of its entities
exist in the 1.1 prototype table, rather than keeping whichever entities are
recognised: a Space Age design imported entity-by-entity arrives as a handful of
belts around holes, and every hole is a lesson. And curved rails are refused
outright — their footprint is not a rectangle, so the tile model genuinely
cannot represent them and every overlap check involving one is wrong in both
directions.

`experiments/real_yield.py` prints that whole table for any cache, through
`real.designs` itself rather than a copy of the filter.

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

Everything in that table is a property of a tidy arrangement, and the run in
issue #17 finished with five of the six at 1.000. `experiments/grader_blindspots.py`
is what that turned out to mean: take module draws the grader calls flawless,
break exactly one thing, and ask again. Over 200 draws —

| perturbation | valid | quality | perfect | flow |
| --- | ---: | ---: | ---: | ---: |
| untouched | 1.000 | 1.000 | 1.000 | 1.000 |
| one assembler retargeted to a recipe the belts do not carry | 1.000 | 1.000 | **1.000** | 0.373 |
| one inserter turned end for end | 1.000 | 1.000 | **1.000** | 0.422 |
| the inserter that takes the product away deleted | 1.000 | 1.000 | **1.000** | 0.422 |

— `perfect` being the fraction scoring 1.0 on *every* metric above. It is 1.000
in all three rows: a factory that provably cannot work scores a perfect mark.
So there is a second tier, which asks whether items can reach the machines
rather than whether the arrangement is legal:

| | |
| --- | --- |
| `fed` | fraction of machines whose ingredients can actually reach them |
| `delivers` | the declared output leaves through the declared port |
| `working` | machines that are both fed and can hand their product on |
| `mixed` | belts carrying more than the two item types a belt holds |
| `leaks` | a belt spilling an item off the edge at no declared port |
| `within zone` | nothing outside the zone the spec asked for |
| `flow` | zero if invalid, else `0.7 delivers + 0.3 fed` |

`flow.py` computes it as a fixed point over item sets, by the game's rules and
not by intuition: an inserter faces the tile it inserts *into* and picks up from
the opposite one, a belt leads into the tile it faces, a belt feeds another belt
and never a machine, and a belt holds at most two item types. It is deliberately
optimistic — throughput, belt sides and furnace fuel are not modelled — because
its job is to separate a factory that works from one that is decorative, which
the table above says the first tier cannot do.

`flow()` is reported *beside* `quality()` and never folded into it. The two
measure different failures, decoration versus a broken chain, and one average
would let either hide the other — which is exactly how the analysed run ended up
with a headline number that had nothing left to say. The whole argument, and
what follows from it, is `docs/FACTORIO.md`.

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
