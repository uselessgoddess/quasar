"""A Factorio blueprint training harness for quasar.

The Rust crate next door trains language models on text. This package turns
Factorio blueprints into text it can train on, and grades what comes back out
the way the game would: does it paste, does it overlap, is it powered, do the
inserters touch anything.

Everything here is dependency-free by design. The corpus, the tokenizer and the
shard files are produced with the standard library alone, so `quasar train` can
be pointed at the output without a Python runtime anywhere near it.
"""

from .blueprint import GRID, Blueprint, BlueprintError, Placement
from .grammar import Spec, parse, serialise, vocabulary
from .prototypes import Data, Entity, Recipe, load
from .validate import Report, Summary, grade, inspect, summarise

__all__ = [
    "GRID",
    "Blueprint",
    "BlueprintError",
    "Data",
    "Entity",
    "Placement",
    "Recipe",
    "Report",
    "Spec",
    "Summary",
    "grade",
    "inspect",
    "load",
    "parse",
    "serialise",
    "summarise",
    "vocabulary",
]
