"""Corpus assembly: what actually lands on disk for the GPU to read."""

import json

import pytest

from quasar_factorio import augment, dataset, grammar, prototypes, shards, tokenizer

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
        assert entry["prompt"] == grammar.prompt(grammar.Spec(**entry["spec"]))
        assert entry["reference"].startswith(entry["prompt"])


def test_a_smaller_corpus_is_a_prefix_of_a_larger_one():
    """Design `n` does not depend on how many designs were asked for."""
    few = [design.spec for design in dataset.designs(5, seed=1, data=DATA)]
    many = [design.spec for design in dataset.designs(12, seed=1, data=DATA)]
    assert many[:5] == few


def test_extra_designs_join_the_mixture(tmp_path):
    """Where scraped human blueprints will arrive, held to the same rules."""
    extra = list(dataset.designs(6, seed=99, data=DATA))
    stats = dataset.build(tmp_path, 4, seed=0, variants=1, extra=iter(extra), data=DATA)
    assert stats.designs + stats.duplicates == 10
    assert stats.designs > 4  # the extras are not silently dropped


def test_a_design_offered_twice_is_only_written_once(tmp_path):
    twice = list(dataset.designs(3, seed=11, data=DATA)) * 2
    stats = dataset.build(tmp_path, 0, variants=2, extra=iter(twice), data=DATA)
    assert (stats.designs, stats.duplicates) == (3, 3)
