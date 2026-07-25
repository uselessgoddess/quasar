"""A WordLevel `tokenizer.json` that quasar loads without a line of Rust changed.

quasar's own tokenizer is a byte-level BPE trained on the corpus. That is right
for prose and wrong here: the corpus is already a token stream, and letting BPE
merge `x1` `2` into some corpus-specific unit throws away the one property worth
having — that every token is a complete, meaningful field.

So the harness emits a WordLevel vocabulary instead. `tokenizers` deserialises it
through the same `Tokenizer::from_file` quasar already calls, and
`Tokenizer::wrap` only requires that `<|endoftext|>` resolve, which it does.

`WhitespaceSplit` rather than `Whitespace`: the latter is a `\\w+|[^\\w\\s]+`
regex and would shred `<|endoftext|>` into five pieces.
"""

from __future__ import annotations

import json
import pathlib

from .grammar import EOS, UNK, vocabulary
from .prototypes import Data


def build(data: Data | None = None) -> dict:
    """The tokenizer document, as a dict.

    `<unk>` is id 0 and `<|endoftext|>` id 1 by construction of the vocabulary
    order, which is worth knowing when reading raw shard bytes in a hex editor.
    """
    tokens = vocabulary(data)
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": tokens.index(token),
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for token in (UNK, EOS)
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        # No decoder: `tokenizers` then joins tokens with a single space, which
        # is exactly the serialisation `grammar.parse` expects to read back.
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {token: index for index, token in enumerate(tokens)},
            "unk_token": UNK,
        },
    }


def write(path: pathlib.Path, data: Data | None = None) -> int:
    """Write `tokenizer.json` and return the vocabulary size."""
    document = build(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n")
    return len(document["model"]["vocab"])


class Encoder:
    """The Python half of the same vocabulary, for building shards.

    Reimplemented rather than imported so the dataset builder has no dependency
    on the `tokenizers` wheel; `tests/test_tokenizer.py` pins the two together
    by round-tripping through the emitted JSON.
    """

    def __init__(self, data: Data | None = None) -> None:
        self.tokens = vocabulary(data)
        self.ids = {token: index for index, token in enumerate(self.tokens)}
        self.unk = self.ids[UNK]
        self.eos = self.ids[EOS]

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.ids.get(token, self.unk) for token in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.tokens[index] for index in ids)
