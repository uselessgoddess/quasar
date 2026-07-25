"""The emitted `tokenizer.json` and the Python encoder have to agree exactly.

They are two implementations of one vocabulary — quasar reads the JSON through
the `tokenizers` crate, the dataset builder writes shards with the Python one —
and a disagreement would put token ids in the shards that mean something else
to the model. Nothing else in the harness would notice.
"""

import json

import pytest

from quasar_factorio import grammar, prototypes, synth, tokenizer

DATA = prototypes.load()


def test_the_document_is_the_shape_the_tokenizers_crate_deserialises():
    document = tokenizer.build(DATA)
    assert document["model"]["type"] == "WordLevel"
    assert document["model"]["unk_token"] == grammar.UNK
    # `Whitespace` is a regex that would shred `<|endoftext|>` into five pieces.
    assert document["pre_tokenizer"] == {"type": "WhitespaceSplit"}
    assert document["decoder"] is None
    specials = {token["content"] for token in document["added_tokens"]}
    assert specials == {grammar.UNK, grammar.EOS}
    assert all(token["special"] for token in document["added_tokens"])


def test_the_special_tokens_sit_at_the_ids_the_shards_assume():
    vocab = tokenizer.build(DATA)["model"]["vocab"]
    assert vocab[grammar.UNK] == 0
    assert vocab[grammar.EOS] == 1


def test_writing_returns_the_vocabulary_size_and_valid_json(tmp_path):
    path = tmp_path / "tokenizer.json"
    size = tokenizer.write(path, DATA)
    document = json.loads(path.read_text())
    assert size == len(document["model"]["vocab"]) == len(grammar.vocabulary(DATA))
    assert path.read_text().endswith("\n")


def test_the_encoder_and_the_emitted_vocabulary_are_the_same_table(tmp_path):
    path = tmp_path / "tokenizer.json"
    tokenizer.write(path, DATA)
    written = json.loads(path.read_text())["model"]["vocab"]
    encoder = tokenizer.Encoder(DATA)
    assert encoder.ids == written


def test_encoding_a_document_round_trips_through_the_ids():
    encoder = tokenizer.Encoder(DATA)
    import random

    blueprint, spec = synth.sample(random.Random(5), DATA)
    text = grammar.serialise(blueprint, DATA, spec)
    ids = encoder.encode(text)
    assert encoder.decode(ids) == text
    assert max(ids) < len(encoder)


def test_an_unknown_token_becomes_unk_rather_than_raising():
    encoder = tokenizer.Encoder(DATA)
    assert encoder.encode("<bp> nonsense-machine </bp>")[1] == encoder.unk


@pytest.mark.parametrize("token", ["<bp>", "</bp>", "<e>", "x00", "y63", "d7", "#64"])
def test_the_structural_tokens_survive_whitespace_splitting(token):
    encoder = tokenizer.Encoder(DATA)
    assert encoder.encode(token) == [encoder.ids[token]]
