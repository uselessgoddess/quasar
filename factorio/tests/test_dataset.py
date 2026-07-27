"""Corpus assembly: what actually lands on disk for the GPU to read."""

import json
import random
from dataclasses import replace

import pytest

from quasar_factorio import (
    augment,
    benchmark,
    dataset,
    grammar,
    prototypes,
    shards,
    synth,
    tokenizer,
    validate,
)

DATA = prototypes.load()


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    stats = dataset.build(out, 120, seed=3, variants=3, valid_every=10, prompts=8, data=DATA)
    return out, stats


def test_the_layout_is_the_one_quasar_train_expects(corpus):
    out, _ = corpus
    assert (out / "train" / "meta.json").exists()
    assert (out / "valid" / "meta.json").exists()
    assert (out / "tokenizer.json").exists()
    assert sorted(path.name for path in (out / "train").glob("*.bin")) == ["shard_0000.bin"]


def test_every_document_survives_the_round_trip_through_the_shards(corpus):
    out, stats = corpus
    encoder = tokenizer.Encoder(DATA)
    tokens, meta = shards.read(out / "train")
    assert meta.vocab_size == len(encoder) == stats.vocab_size
    assert meta.eos == encoder.eos
    assert meta.docs == stats.train_docs
    documents = list(shards.documents(tokens, meta.eos))
    assert len(documents) == meta.docs
    for ids in documents:
        blueprint, spec = grammar.parse(encoder.decode(ids), DATA)
        assert blueprint.entities and spec is not None


def test_no_token_lands_outside_the_vocabulary(corpus):
    out, stats = corpus
    tokens, _ = shards.read(out / "train")
    assert max(tokens) < stats.vocab_size


def test_no_document_is_written_twice(corpus):
    out, _ = corpus
    encoder = tokenizer.Encoder(DATA)
    seen = set()
    for split in ("train", "valid"):
        tokens, meta = shards.read(out / split)
        for ids in shards.documents(tokens, meta.eos):
            text = encoder.decode(ids)
            assert text not in seen
            seen.add(text)


def test_a_design_never_straddles_the_split(corpus):
    """The point of splitting by design: no mirror image leaks into `valid`.

    Compared through `augment.canonical`, so a validation blueprint that appears
    in training rotated, reflected or upgraded to red belts is caught — which is
    the only comparison that proves the validation loss is measuring
    generalisation rather than recall.
    """
    out, _ = corpus
    encoder = tokenizer.Encoder(DATA)

    def keys(split):
        tokens, meta = shards.read(out / split)
        found = set()
        for ids in shards.documents(tokens, meta.eos):
            blueprint, _ = grammar.parse(encoder.decode(ids), DATA)
            found.add(augment.canonical(blueprint, DATA))
        return found

    assert not keys("train") & keys("valid")


def test_the_manifest_adds_up(corpus):
    out, stats = corpus
    written = json.loads((out / "manifest.json").read_text())
    assert written == stats.to_dict()
    assert stats.rejected == 0
    assert stats.documents == stats.train_docs + stats.valid_docs
    assert stats.valid_docs > 0
    assert sum(stats.kinds.values()) == stats.designs
    # Every document plus its terminator has to fit the context the nano
    # preset trains at; a truncated document teaches `</bp>` is optional.
    assert stats.longest <= 512


def test_prompts_are_held_out_and_replayable(corpus):
    out, _ = corpus
    prompts = dataset.read_prompts(out / "prompts.jsonl")
    assert 0 < len(prompts) <= 8
    encoder = tokenizer.Encoder(DATA)
    tokens, meta = shards.read(out / "train")
    trained = {encoder.decode(ids) for ids in shards.documents(tokens, meta.eos)}
    for entry in prompts:
        assert entry["reference"] not in trained
        # Through the JSON and back: a spec whose ports survive as plain dicts
        # would produce a prompt with no ports in it, which is a different task
        # than the one the reference answers.
        assert entry["prompt"] == grammar.prompt(grammar.Spec.from_dict(entry["spec"]))
        assert entry["reference"].startswith(entry["prompt"])


def test_the_fixed_module_benchmark_is_stratified_and_working(corpus):
    """The primary metric has 64 real denominators, not five lucky modules."""
    out, stats = corpus
    records = dataset.read_prompts(out / "benchmark.jsonl")
    counts = {}
    shapes = {}
    for record in records:
        counts[record["benchmark_target"]] = counts.get(record["benchmark_target"], 0) + 1
        shapes[record["benchmark_target"]] = record["shape"]
        assert record["benchmark"] == benchmark.VERSION
        assert record["kind"] == "module"
        assert record["reference"].startswith(record["prompt"])
        report = validate.grade(record["reference"], DATA)
        assert (report.delivers, report.fed, report.working) == (1.0, 1.0, 1.0)

    assert len(records) == stats.benchmark_prompts == 64
    assert len(counts) == len(benchmark.TARGETS) == 29
    assert min(counts.values()) >= 2
    assert all(counts[name] >= 3 for name, shape in shapes.items() if shape == "fork")


def test_the_fixed_benchmark_does_not_depend_on_the_corpus_seed(tmp_path):
    """Changing training data must not move the evaluation goalposts."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    dataset.build(first, 12, seed=1, variants=1, data=DATA)
    dataset.build(second, 12, seed=999, variants=1, data=DATA)
    assert (first / "benchmark.jsonl").read_bytes() == (second / "benchmark.jsonl").read_bytes()


def test_no_fixed_benchmark_layout_is_written_to_either_shard(corpus):
    """Exact reference geometry is reserved before the split is assigned."""
    out, _ = corpus
    reserved = {
        augment.canonical(grammar.parse(record["reference"], DATA)[0], DATA)
        for record in dataset.read_prompts(out / "benchmark.jsonl")
    }
    encoder = tokenizer.Encoder(DATA)
    written = set()
    for split in ("train", "valid"):
        tokens, meta = shards.read(out / split)
        for ids in shards.documents(tokens, meta.eos):
            blueprint, _ = grammar.parse(encoder.decode(ids), DATA)
            written.add(augment.canonical(blueprint, DATA))
    assert not reserved & written


def test_the_first_prompts_are_one_of_each_kind_rather_than_a_slice_of_the_mixture(tmp_path):
    """What a caller taking a prefix gets, which is every caller there is.

    Sampling is the expensive half of a run, so `examples/factorio.sh` grades
    the first two dozen prompts and nothing else. In stream order those two
    dozen are a sample of the mixture — mostly belt lanes and assembler rows,
    the kinds whose metrics already read 1.000 — and the module prompts, the
    only ones that ask whether items can reach a machine, would be a couple of
    entries at best.
    """
    out = tmp_path / "corpus"
    dataset.build(out, 400, seed=5, variants=2, valid_every=6, prompts=8, data=DATA)
    kinds = [entry["kind"] for entry in dataset.read_prompts(out / "prompts.jsonl")]
    assert len(kinds) == len(set(kinds))


def test_a_smaller_corpus_is_a_prefix_of_a_larger_one():
    """Design `n` does not depend on how many designs were asked for."""
    few = [design.spec for design in dataset.designs(5, seed=1, data=DATA)]
    many = [design.spec for design in dataset.designs(12, seed=1, data=DATA)]
    assert many[:5] == few


def test_extra_designs_join_the_mixture(tmp_path):
    """Where scraped human blueprints will arrive, held to the same rules."""
    extra = list(dataset.designs(6, seed=99, data=DATA))
    stats = dataset.build(tmp_path, 4, seed=0, variants=1, extra=iter(extra), data=DATA)
    assert stats.designs + stats.repeats + stats.benchmark_excluded == 10
    assert stats.designs > 4  # the extras are not silently dropped


def test_a_design_offered_twice_is_only_written_once(tmp_path):
    twice = list(dataset.designs(3, seed=11, data=DATA)) * 2
    stats = dataset.build(tmp_path, 0, variants=2, extra=iter(twice), data=DATA)
    assert (stats.designs, stats.repeats) == (3, 3)
    # Every document of the second pass is byte-identical to one of the first.
    assert stats.duplicates == stats.documents


def test_a_repeated_layout_keeps_the_documents_that_are_new(tmp_path):
    """The case the merge exists for: same layout, different spec.

    A smelter column for copper is the iron one with one token changed, so
    `augment.canonical` calls them one design — but the prompt differs, and
    `r:` is the whole conditioning signal. Dropping the second would teach the
    model that the requested product does not constrain the answer.
    """
    blueprint, spec = synth.GENERATORS["smelter-column"](random.Random(0), DATA)
    # A furnace carries no recipe token, so the product lives only in the spec —
    # which is exactly why the two documents share a canonical key.
    assert spec.product in {"iron-plate", "copper-plate", "steel-plate", "stone-brick"}

    def design(product):
        made = replace(spec, product=product)
        return dataset.Design(
            kind=made.kind,
            blueprint=blueprint,
            spec=made,
            documents=[grammar.serialise(blueprint, DATA, made)],
        )

    iron, copper = design("iron-plate"), design("stone-brick")
    assert iron.documents != copper.documents  # only the spec line differs

    stats = dataset.build(tmp_path, 0, extra=iter([iron, copper]), data=DATA)
    assert (stats.designs, stats.repeats, stats.duplicates, stats.rejected) == (1, 1, 0, 0)
    assert stats.documents == 2


def test_a_repeated_layout_does_not_straddle_the_split(tmp_path):
    """Merged documents follow the layout, so neither split can learn the other."""
    held = next(iter(dataset.designs(1, seed=5, data=DATA)))
    stats = dataset.build(tmp_path, 0, extra=iter([held, held, held]), data=DATA)
    # Design 0 is always the held-out one, and the repeats have to follow it.
    assert (stats.train_docs, stats.designs, stats.repeats) == (0, 1, 2)
    assert stats.valid_docs > 0
