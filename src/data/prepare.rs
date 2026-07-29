//! The one-off pass that turns a corpus into a shard directory.
//!
//! Runs once per corpus and then never again, so it is written for throughput:
//! documents are tokenised a chunk at a time across every core, which is the
//! only part of the pipeline that is CPU-bound.

use std::io;
use std::path::Path;
use std::vec;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::data::corpus;
use crate::data::shard::{self, Meta};
use crate::data::tokenizer;
use crate::data::{Corpus, Tokenizer};

/// Documents tokenised per parallel chunk. Large enough to amortise the join,
/// small enough that a chunk of FineWeb-Edu documents stays under a gigabyte.
const CHUNK: usize = 4096;

/// Per-source chunk for a mixture. Keeping one encoded chunk buffered for each
/// source makes weighted scheduling cheap without holding four full [`CHUNK`]s
/// of long web documents in memory at once.
const MIX_CHUNK: usize = 256;

/// One document in `VALID_EVERY` is held out. At 1 in 200 a 10B-token corpus
/// leaves ~50M validation tokens — far more than the eval loop needs, and the
/// split is by whole document, so no validation text can appear in training.
const VALID_EVERY: u64 = 200;

#[derive(Debug)]
pub enum Error {
    Corpus(corpus::Error),
    Tokenizer(tokenizer::Error),
    Io(io::Error),
    InvalidMix(&'static str),
    /// A finite source cannot supply its requested share of the token budget.
    Exhausted {
        source: usize,
        train_tokens: u64,
        requested_tokens: u64,
    },
}

/// What `prepare` wrote, one [`Meta`] per split.
#[derive(Debug)]
pub struct Prepared {
    pub train: Meta,
    pub valid: Meta,
    /// Per-source counts. A plain [`run`] has one entry; [`mix`] has one in the
    /// same order as its [`MixSource`] arguments.
    pub sources: Vec<SourcePrepared>,
}

/// A corpus and its relative token weight.
///
/// Weights do not have to sum to 100. `3:1`, `75:25`, and `0.75:0.25` after a
/// caller's fixed-point parsing all describe the same mixture.
#[derive(Debug, Clone, Copy)]
pub struct MixSource<'a> {
    pub corpus: &'a Corpus,
    pub weight: u64,
}

/// What one source contributed to the two output shards.
#[derive(Debug, Clone, Copy, Default)]
pub struct SourcePrepared {
    pub train_tokens: u64,
    pub train_docs: u64,
    pub train_bytes: u64,
    pub valid_tokens: u64,
    pub valid_docs: u64,
    pub valid_bytes: u64,
}

/// Tokenise every document of `corpus` into `out/train` and `out/valid`.
pub fn run(corpus: &Corpus, tokenizer: &Tokenizer, out: &Path) -> Result<Prepared, Error> {
    let (vocab, eos) = (tokenizer.vocab_size(), tokenizer.eos());
    let mut train = shard::Writer::create(&out.join("train"), vocab, eos).map_err(Error::Io)?;
    let mut valid = shard::Writer::create(&out.join("valid"), vocab, eos).map_err(Error::Io)?;

    let bar = ProgressBar::new_spinner().with_style(style());
    let mut index = 0u64;
    for chunk in chunks(corpus.docs()) {
        let encoded: Vec<_> = chunk?
            .par_iter()
            .map(|doc| tokenizer.encode(doc).map(|ids| (ids, doc.len())))
            .collect::<Result<_, _>>()
            .map_err(Error::Tokenizer)?;

        for (ids, bytes) in encoded {
            let writer = if index.is_multiple_of(VALID_EVERY) { &mut valid } else { &mut train };
            writer.push(&ids, bytes).map_err(Error::Io)?;
            index += 1;
        }
        bar.set_message(format!("{index} docs"));
        bar.tick();
    }
    bar.finish();

    let train = train.finish().map_err(Error::Io)?;
    let valid = valid.finish().map_err(Error::Io)?;
    Ok(Prepared {
        train: train.clone(),
        valid: valid.clone(),
        sources: vec![SourcePrepared {
            train_tokens: train.tokens,
            train_docs: train.docs,
            train_bytes: train.bytes,
            valid_tokens: valid.tokens,
            valid_docs: valid.docs,
            valid_bytes: valid.bytes,
        }],
    })
}

/// Build at least `train_tokens` of training shards at the requested token
/// proportions.
///
/// Scheduling happens after tokenisation: the trainer draws uniform token
/// offsets, so document counts and compressed download sizes are not the
/// distribution it sees. Whole documents are kept intact, which means the
/// final budget may be exceeded by one document and each source's share may
/// differ by at most roughly one document.
///
/// Every source gets its own 1-in-[`VALID_EVERY`] document split. A global
/// counter would phase-lock a periodic weighted schedule and could put every
/// validation document into the same source.
pub fn mix(
    sources: &[MixSource<'_>],
    tokenizer: &Tokenizer,
    out: &Path,
    train_tokens: u64,
) -> Result<Prepared, Error> {
    if sources.len() < 2 {
        return Err(Error::InvalidMix("a mixture needs at least two sources"));
    }
    if train_tokens == 0 {
        return Err(Error::InvalidMix("the training token budget must be positive"));
    }
    if sources.iter().any(|source| source.weight == 0) {
        return Err(Error::InvalidMix("mixture weights must be positive"));
    }

    let (vocab, eos) = (tokenizer.vocab_size(), tokenizer.eos());
    let mut train = shard::Writer::create(&out.join("train"), vocab, eos).map_err(Error::Io)?;
    let mut valid = shard::Writer::create(&out.join("valid"), vocab, eos).map_err(Error::Io)?;
    let mut streams: Vec<_> =
        sources.iter().map(|source| EncodedDocs::new(source.corpus, tokenizer)).collect();
    let mut prepared = vec![SourcePrepared::default(); sources.len()];
    let mut seen = vec![0u64; sources.len()];

    let bar = ProgressBar::new(train_tokens).with_style(mix_style());
    let mut total = 0u64;
    while total < train_tokens {
        let source = lightest(sources, &prepared);
        let Some((ids, bytes)) = streams[source].next()? else {
            return Err(Error::Exhausted {
                source,
                train_tokens: prepared[source].train_tokens,
                requested_tokens: train_tokens,
            });
        };

        let report = &mut prepared[source];
        if seen[source].is_multiple_of(VALID_EVERY) {
            valid.push(&ids, bytes).map_err(Error::Io)?;
            report.valid_tokens += ids.len() as u64;
            report.valid_docs += 1;
            report.valid_bytes += bytes as u64;
        } else {
            train.push(&ids, bytes).map_err(Error::Io)?;
            report.train_tokens += ids.len() as u64;
            report.train_docs += 1;
            report.train_bytes += bytes as u64;
            total += ids.len() as u64;
            bar.set_position(total.min(train_tokens));
        }
        seen[source] += 1;
    }
    bar.finish();

    Ok(Prepared {
        train: train.finish().map_err(Error::Io)?,
        valid: valid.finish().map_err(Error::Io)?,
        sources: prepared,
    })
}

/// Source furthest below its relative allocation, compared without floats.
fn lightest(sources: &[MixSource<'_>], prepared: &[SourcePrepared]) -> usize {
    (0..sources.len())
        .min_by(|&left, &right| {
            let left_scaled = prepared[left].train_tokens as u128 * sources[right].weight as u128;
            let right_scaled = prepared[right].train_tokens as u128 * sources[left].weight as u128;
            left_scaled.cmp(&right_scaled).then_with(|| left.cmp(&right))
        })
        .expect("mix validation requires two sources")
}

type Docs<'a> = Box<dyn Iterator<Item = Result<String, corpus::Error>> + 'a>;

/// Lazily tokenises one small parallel chunk from one mixture source.
struct EncodedDocs<'a> {
    docs: Docs<'a>,
    tokenizer: &'a Tokenizer,
    buffered: vec::IntoIter<(Vec<u16>, usize)>,
}

impl<'a> EncodedDocs<'a> {
    fn new(corpus: &'a Corpus, tokenizer: &'a Tokenizer) -> Self {
        Self { docs: Box::new(corpus.docs()), tokenizer, buffered: Vec::new().into_iter() }
    }

    fn next(&mut self) -> Result<Option<(Vec<u16>, usize)>, Error> {
        if let Some(encoded) = self.buffered.next() {
            return Ok(Some(encoded));
        }

        let docs: Vec<_> =
            self.docs.by_ref().take(MIX_CHUNK).collect::<Result<_, _>>().map_err(Error::Corpus)?;
        if docs.is_empty() {
            return Ok(None);
        }
        let encoded: Vec<_> = docs
            .par_iter()
            .map(|doc| self.tokenizer.encode(doc).map(|ids| (ids, doc.len())))
            .collect::<Result<_, _>>()
            .map_err(Error::Tokenizer)?;
        self.buffered = encoded.into_iter();
        Ok(self.buffered.next())
    }
}

/// Group a fallible document stream into owned chunks, failing the whole chunk
/// on the first bad document.
fn chunks<I>(docs: I) -> impl Iterator<Item = Result<Vec<String>, Error>>
where
    I: Iterator<Item = Result<String, corpus::Error>>,
{
    let mut docs = docs.fuse();
    std::iter::from_fn(move || {
        let chunk: Result<Vec<String>, _> = docs.by_ref().take(CHUNK).collect();
        match chunk {
            Ok(chunk) if chunk.is_empty() => None,
            Ok(chunk) => Some(Ok(chunk)),
            Err(error) => Some(Err(Error::Corpus(error))),
        }
    })
}

fn style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner} tokenising {msg} in {elapsed}").unwrap()
}

fn mix_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner} mixing [{bar:32}] {pos}/{len} train tokens in {elapsed}",
    )
    .unwrap()
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Corpus(error) => write!(f, "{error}"),
            Self::Tokenizer(error) => write!(f, "{error}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::InvalidMix(message) => write!(f, "{message}"),
            Self::Exhausted { source, train_tokens, requested_tokens } => write!(
                f,
                "mixture source #{} ended after {train_tokens} training tokens, \
                 before the mixture could reach its {requested_tokens}-token budget",
                source + 1
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Shards;

    fn corpus(docs: &[&str]) -> (tempfile::TempDir, Corpus) {
        let dir = tempfile::tempdir().unwrap();
        let lines: String =
            docs.iter().map(|d| format!("{{\"text\":\"{d}\"}}\n")).collect::<Vec<_>>().concat();
        std::fs::write(dir.path().join("a.jsonl"), lines).unwrap();
        let corpus = Corpus::open(&[dir.path().to_owned()], "text").unwrap();
        (dir, corpus)
    }

    fn tokenizer() -> Tokenizer {
        Tokenizer::train(["the quick brown fox jumps"].into_iter(), 300).unwrap()
    }

    #[test]
    fn every_document_lands_in_exactly_one_split() {
        let (_dir, corpus) = corpus(&["one", "two", "three"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        assert_eq!(prepared.train.docs + prepared.valid.docs, 3);
    }

    #[test]
    fn the_first_document_is_held_out() {
        let (_dir, corpus) = corpus(&["one", "two", "three"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        assert_eq!(prepared.valid.docs, 1);
    }

    #[test]
    fn the_shards_are_readable_afterwards() {
        let (_dir, corpus) = corpus(&["the quick brown fox", "jumps over"]);
        let out = tempfile::tempdir().unwrap();

        let prepared = run(&corpus, &tokenizer(), out.path()).unwrap();

        let shards = Shards::open(&out.path().join("train")).unwrap();
        assert_eq!(shards.len() as u64, prepared.train.tokens);
    }

    /// Issue #25: passing two directories to the old `prepare` concatenated
    /// them, so a 3:1 request became whatever ratio their downloaded sizes
    /// happened to have. A mixture is measured after tokenisation because
    /// uniform offsets in [`crate::data::Batcher`] make shard tokens the
    /// training distribution.
    #[test]
    fn a_mix_follows_token_weights_instead_of_source_sizes() {
        let first: Vec<_> =
            (0..120).map(|i| format!("alpha lesson number {i} with equal length")).collect();
        let second: Vec<_> =
            (0..120).map(|i| format!("omega lesson number {i} with equal length")).collect();
        let first_refs: Vec<_> = first.iter().map(String::as_str).collect();
        let second_refs: Vec<_> = second.iter().map(String::as_str).collect();
        let tokenizer =
            Tokenizer::train(first_refs.iter().chain(&second_refs).copied(), 400).unwrap();
        let (_first_dir, first) = corpus(&first_refs);
        let (_second_dir, second) = corpus(&second_refs);
        let out = tempfile::tempdir().unwrap();

        let prepared = mix(
            &[MixSource { corpus: &first, weight: 3 }, MixSource { corpus: &second, weight: 1 }],
            &tokenizer,
            out.path(),
            1_000,
        )
        .unwrap();

        let total: u64 = prepared.sources.iter().map(|source| source.train_tokens).sum();
        let first = prepared.sources[0].train_tokens as f64 / total as f64;
        assert!((first - 0.75).abs() < 0.03, "wanted 75%, prepared {:.2}%", first * 100.0);
        assert!(prepared.train.tokens >= 1_000, "a token budget is a lower bound");
    }
}
