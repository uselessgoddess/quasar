"""Writing quasar's shard format from Python.

The format is deliberately small: little-endian `u16` per token, concatenated
documents, rotated every 512 MB, plus a `meta.json` sidecar. Emitting it here
rather than shelling out to `quasar prepare` skips a JSONL round trip through a
byte-level BPE that this corpus does not want, and keeps the token ids the
dataset builder chose all the way to the GPU.

Mirrors `src/data/shard.rs`; `tests/test_shards.py` and the Rust side's own
tests both assert the same layout.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass

SHARD_BYTES = 512 << 20

# `src/data/prepare.rs` holds out one document in every 200 for validation. Same
# constant, same stride, so a corpus built here splits the way a corpus built by
# `quasar prepare` would.
VALID_EVERY = 200


@dataclass
class Meta:
    tokens: int = 0
    docs: int = 0
    bytes: int = 0
    vocab_size: int = 0
    eos: int = 0


def shard_path(directory: pathlib.Path, index: int) -> pathlib.Path:
    return directory / f"shard_{index:04d}.bin"


class Writer:
    """Append documents, then :meth:`finish` to flush and write `meta.json`."""

    def __init__(self, directory: pathlib.Path, vocab_size: int, eos: int) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.in_shard = 0
        self.file = shard_path(directory, 0).open("wb")
        self.meta = Meta(vocab_size=vocab_size, eos=eos)

    def push(self, ids: list[int], text_bytes: int) -> None:
        """One document. The terminator is added here, not by the caller.

        `text_bytes` is the UTF-8 length of the document as written, which is
        what makes quasar's bits-per-byte figure comparable between this corpus
        and a natural-language one.
        """
        ids = [*ids, self.meta.eos]
        blob = b"".join(value.to_bytes(2, "little") for value in ids)
        self.file.write(blob)
        self.in_shard += len(blob)
        self.meta.tokens += len(ids)
        self.meta.docs += 1
        self.meta.bytes += text_bytes
        if self.in_shard >= SHARD_BYTES:
            self._rotate()

    def _rotate(self) -> None:
        self.file.close()
        self.index += 1
        self.in_shard = 0
        self.file = shard_path(self.directory, self.index).open("wb")

    def finish(self) -> Meta:
        self.file.close()
        if self.in_shard == 0 and self.index > 0:
            shard_path(self.directory, self.index).unlink()
        (self.directory / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        return self.meta


def read(directory: pathlib.Path) -> tuple[list[int], Meta]:
    """Read a shard directory back. For tests and for inspecting a corpus."""
    meta = Meta(**json.loads((directory / "meta.json").read_text()))
    tokens: list[int] = []
    for path in sorted(directory.glob("*.bin")):
        blob = path.read_bytes()
        tokens += [int.from_bytes(blob[at : at + 2], "little") for at in range(0, len(blob), 2)]
    return tokens, meta


def documents(tokens: list[int], eos: int):
    """Split a flat token stream back into documents."""
    current: list[int] = []
    for token in tokens:
        if token == eos:
            yield current
            current = []
        else:
            current.append(token)
    if current:
        yield current
