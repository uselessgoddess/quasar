"""The on-disk format quasar's data loader memory-maps.

`src/data/shard.rs` reads these files; nothing in Python will ever notice if the
byte order flips or a terminator goes missing, so the layout is asserted here
against literal bytes rather than against the reader in the same module.
"""

import json

from quasar_factorio import shards


def test_a_document_is_little_endian_u16_with_a_terminator(tmp_path):
    writer = shards.Writer(tmp_path, vocab_size=300, eos=1)
    writer.push([7, 258], text_bytes=11)
    writer.finish()
    blob = (tmp_path / "shard_0000.bin").read_bytes()
    # 7, 258, EOS — 258 is the byte pair that would look identical big-endian.
    assert blob == b"\x07\x00\x02\x01\x01\x00"


def test_the_meta_sidecar_carries_what_the_loader_reads(tmp_path):
    writer = shards.Writer(tmp_path, vocab_size=495, eos=1)
    writer.push([2, 3, 4], text_bytes=20)
    writer.push([5], text_bytes=6)
    meta = writer.finish()
    written = json.loads((tmp_path / "meta.json").read_text())
    assert written == {"tokens": 6, "docs": 2, "bytes": 26, "vocab_size": 495, "eos": 1}
    assert written == {
        "tokens": meta.tokens,
        "docs": meta.docs,
        "bytes": meta.bytes,
        "vocab_size": meta.vocab_size,
        "eos": meta.eos,
    }


def test_the_terminator_is_added_here_and_not_by_the_caller(tmp_path):
    writer = shards.Writer(tmp_path, vocab_size=10, eos=1)
    writer.push([9], text_bytes=1)
    meta = writer.finish()
    tokens, _ = shards.read(tmp_path)
    assert tokens == [9, 1]
    assert meta.tokens == 2


def test_reading_back_concatenates_shards_in_order(tmp_path):
    writer = shards.Writer(tmp_path, vocab_size=10, eos=1)
    writer.push([2], text_bytes=1)
    # Force the rotation the 512 MB threshold would eventually cause.
    writer._rotate()
    writer.push([3], text_bytes=1)
    writer.finish()
    assert sorted(path.name for path in tmp_path.glob("*.bin")) == [
        "shard_0000.bin",
        "shard_0001.bin",
    ]
    tokens, meta = shards.read(tmp_path)
    assert tokens == [2, 1, 3, 1]
    assert list(shards.documents(tokens, meta.eos)) == [[2], [3]]


def test_an_empty_trailing_shard_is_not_left_behind(tmp_path):
    """A zero-byte `.bin` would make the Rust loader map an empty file."""
    writer = shards.Writer(tmp_path, vocab_size=10, eos=1)
    writer.push([2], text_bytes=1)
    writer._rotate()
    writer.finish()
    assert [path.name for path in tmp_path.glob("*.bin")] == ["shard_0000.bin"]


def test_an_empty_split_still_produces_a_readable_directory(tmp_path):
    """`quasar train` opens `valid/` unconditionally, even if nothing held out."""
    meta = shards.Writer(tmp_path, vocab_size=10, eos=1).finish()
    assert meta.docs == 0
    tokens, read_back = shards.read(tmp_path)
    assert tokens == []
    assert read_back == meta


def test_splitting_a_stream_does_not_invent_a_trailing_document():
    assert list(shards.documents([2, 3, 1, 4, 1], 1)) == [[2, 3], [4]]
    # An unterminated tail is still a document — the last write may have been cut off.
    assert list(shards.documents([2, 3, 1, 4], 1)) == [[2, 3], [4]]
    assert list(shards.documents([], 1)) == []


def test_the_shard_naming_matches_the_rust_side(tmp_path):
    """`src/data/shard.rs` globs `shard_%04d.bin` and sorts lexicographically."""
    assert shards.SHARD_BYTES == 512 * 1024 * 1024
    assert shards.shard_path(tmp_path, 0).name == "shard_0000.bin"
    assert shards.shard_path(tmp_path, 42).name == "shard_0042.bin"
    # Zero padding is what keeps the lexicographic sort numeric past nine.
    assert shards.shard_path(tmp_path, 9).name < shards.shard_path(tmp_path, 10).name
