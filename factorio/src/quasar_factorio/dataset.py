"""Building the corpus quasar trains on, end to end.

Draw designs, augment each into its symmetries, serialise, tokenise, and write
`train/` and `valid/` shard directories plus the `tokenizer.json` that reads
them — the same layout `quasar prepare` produces, so `quasar train --data` needs
no argument it did not already have.

Two decisions here are worth more than the rest of the file:

*Nothing invalid gets in.* Every document is graded before it is written, and a
design the grader rejects is dropped rather than repaired. The generators are
held to a perfect score by their own tests, so a rejection means a bug, and the
`rejected` count in the manifest is the alarm for it.

*The split is by design, not by document.* Holding out every Nth *document*
would put a smelter column in `train` and its mirror image in `valid`, and the
validation loss would then measure memorisation. Whole designs move together,
augmentations and all — and "the same design" is decided by
`augment.canonical`, which sees through rotation, reflection and belt tier.

The corollary is that a repeated layout is *merged* into the design already
kept rather than dropped, because the same layout under a different spec is a
different thing to learn. See `build`.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from itertools import zip_longest

from . import augment, grammar, shards, synth, tokenizer, validate
from .blueprint import Blueprint
from .grammar import Spec
from .prototypes import Data, load

# One design in every `VALID_EVERY` is held out. Denser than quasar's own 1-in-200
# because this corpus is small: a nano model's eval loop wants a few hundred
# thousand validation tokens, and at 1-in-200 a 40k-design corpus would not have
# them.
VALID_EVERY = 25


@dataclass
class Design:
    """One drawn layout and the documents it expands into."""

    kind: str
    blueprint: Blueprint
    spec: Spec
    documents: list[str] = field(default_factory=list)


@dataclass
class Stats:
    designs: int = 0
    documents: int = 0
    #: Draws that turned out to be another form of a design already kept. Their
    #: documents are merged into it rather than dropped; see `build`.
    repeats: int = 0
    #: Documents discarded as byte-identical to one already written.
    duplicates: int = 0
    rejected: int = 0
    train_docs: int = 0
    valid_docs: int = 0
    train_tokens: int = 0
    valid_tokens: int = 0
    vocab_size: int = 0
    longest: int = 0
    kinds: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def designs(
    count: int,
    *,
    seed: int = 0,
    data: Data | None = None,
    variants: int = 4,
) -> Iterator[Design]:
    """`count` drawn designs, each expanded into up to `variants` documents.

    The per-design RNG is seeded from the design index rather than shared, so a
    corpus of 10,000 designs contains the first 1,000 of a corpus of 1,000 —
    which makes a small run an honest preview of a large one.
    """
    data = data or load()
    for index in range(count):
        rng = random.Random(seed * 1_000_003 + index)
        blueprint, spec = synth.sample(rng, data)
        design = Design(kind=spec.kind, blueprint=blueprint, spec=spec)
        for form, form_spec in augment.variants(blueprint, spec, data, rng=rng, limit=variants):
            design.documents.append(grammar.serialise(form, data, form_spec))
        yield design


def build(
    out: pathlib.Path,
    count: int = 20_000,
    *,
    seed: int = 0,
    variants: int = 4,
    data: Data | None = None,
    extra: Iterator[Design] | None = None,
    valid_every: int = VALID_EVERY,
    prompts: int = 256,
) -> Stats:
    """Write `out/train`, `out/valid`, `out/tokenizer.json` and `out/manifest.json`.

    `extra` is where scraped human blueprints join the mixture; they arrive as
    `Design`s so they are deduplicated, graded and split by exactly the same
    rules the synthetic ones are.

    Deduplication is by *document*, and the split is by *layout*: a draw whose
    layout has been seen before contributes whatever documents are new to the
    side that layout already sits on. Nothing a generator says twice is written
    twice, and nothing it says once is thrown away for resembling something.
    """
    data = data or load()
    out.mkdir(parents=True, exist_ok=True)
    encoder = tokenizer.Encoder(data)
    tokenizer.write(out / "tokenizer.json", data)

    train = shards.Writer(out / "train", len(encoder), encoder.eos)
    valid = shards.Writer(out / "valid", len(encoder), encoder.eos)
    stats = Stats(vocab_size=len(encoder))
    sides: dict[str, bool] = {}
    seen: set[bytes] = set()
    held: list[Design] = []

    stream = designs(count, seed=seed, data=data, variants=variants)
    for design in _chain(stream, extra):
        # Two draws that differ only by a turn, a flip or a belt tier are one
        # design; splitting them apart would put one in each split. See
        # `augment.canonical`.
        key = augment.canonical(design.blueprint, data)
        if key in sides:
            # Merged, not dropped. A smelter column for copper has the same
            # layout as one for iron and so the same key, but a different
            # prompt — and `r:` is the conditioning signal the model is
            # supposed to learn. Discarding the second one would teach it that
            # the requested product does not matter. It goes to whichever side
            # the layout is already on, which is what keeps the split honest.
            stats.repeats += 1
            holdout = sides[key]
        else:
            stats.designs += 1
            stats.kinds[design.kind] = stats.kinds.get(design.kind, 0) + 1
            # Whole design to one side or the other; see the module docstring.
            holdout = sides[key] = (stats.designs - 1) % valid_every == 0
        if holdout:
            held.append(design)
        for text in dict.fromkeys(design.documents):
            # Hashed rather than kept: a 20k-design corpus is 80k documents,
            # and the texts themselves are the largest thing in the process.
            digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
            if digest in seen:
                stats.duplicates += 1
                continue
            seen.add(digest)
            if not validate.grade(text, data).valid:
                stats.rejected += 1
                continue
            ids = encoder.encode(text)
            (valid if holdout else train).push(ids, len(text.encode()))
            stats.documents += 1
            stats.longest = max(stats.longest, len(ids) + 1)
            if holdout:
                stats.valid_docs += 1
            else:
                stats.train_docs += 1

    stats.train_tokens = train.finish().tokens
    stats.valid_tokens = valid.finish().tokens
    _write_prompts(out / "prompts.jsonl", held, data, prompts)
    (out / "manifest.json").write_text(json.dumps(stats.to_dict(), indent=2) + "\n")
    return stats


def _chain(*streams) -> Iterator[Design]:
    for stream in streams:
        yield from stream or ()


def _write_prompts(path: pathlib.Path, held: list[Design], data: Data, limit: int) -> None:
    """The evaluation set: specs the model has never been trained on.

    Drawn from the held-out designs so that scoring a generation against its
    prompt asks the model to build something new, not to recall something it
    has already written down. The reference document rides along so the
    renderer can put the two side by side.

    Round-robin over kinds rather than in stream order. Callers take a prefix —
    `examples/factorio.sh` samples the first two dozen, because generation is
    the expensive half of the run — and in stream order a prefix is a sample of
    the *mixture*, which is nine parts belt lane and assembler row. The metric
    that still has somewhere to go is item flow, and item flow is a question
    only the module prompts ask, so a prefix that mirrors the mixture spends its
    budget on the part of the benchmark that already reads 1.000.
    """
    by_kind: dict[str, list[Design]] = {}
    for design in held:
        by_kind.setdefault(design.kind, []).append(design)
    ordered = []
    for row in zip_longest(*by_kind.values()):
        ordered.extend(design for design in row if design is not None)

    lines = []
    for design in ordered[:limit]:
        lines.append(
            json.dumps(
                {
                    "kind": design.kind,
                    "spec": design.spec.to_dict(),
                    "prompt": grammar.prompt(design.spec),
                    "reference": design.documents[0] if design.documents else "",
                },
                sort_keys=True,
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_prompts(path: pathlib.Path) -> list[dict]:
    """The eval set back, for the sampler and the plots."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
