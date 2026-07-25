//! The Factorio family: models that write blueprints instead of prose.
//!
//! Kept apart from the language presets because almost nothing they share is a
//! coincidence worth generalising. `tiny` and `base` are sized by a scaling law
//! measured over a 32,768-token BPE vocabulary and a 2,048-token context;
//! these are sized by a corpus whose vocabulary is 495 fixed grammar tokens and
//! whose longest sentence is one 64-entity blueprint. Their knobs move for
//! different reasons and are checked against different things — the tests below
//! compare a member of this family against the corpus `quasar_factorio` writes,
//! not against a parameter class.
//!
//! One member so far. The family exists so the next one — a wider grid, a
//! longer context, circuit wires — is a variant here rather than another
//! preset wedged in among the language models.

use super::Model;

/// Tokens in the blueprint grammar.
///
/// Fixed, not learned: `quasar_factorio.tokenizer` derives the vocabulary from
/// the prototype table, so it changes only when Factorio's own entity list
/// does. `factorio/tests/test_tokenizer.py` pins the same number from the other
/// side, and `quasar train` reads the corpus's real vocabulary anyway — this is
/// what the budget arithmetic assumes, not what the run trusts.
pub const VOCAB: usize = 495;

/// The context one blueprint needs.
///
/// A 64-entity design serialises to about 460 tokens, so 512 holds a whole
/// document with its prompt and there is nothing past it worth attending to.
pub const SEQ_LEN: usize = 512;

/// A member of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    /// 3.5M — the blueprint model the harness trains.
    Nano,
}

impl Size {
    /// Every member, for tests and for `--help` text that cannot drift.
    pub const ALL: [Self; 1] = [Self::Nano];

    pub fn model(self) -> Model {
        match self {
            Self::Nano => nano(),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Nano => "factorio-nano",
        }
    }
}

/// `quasar-factorio-nano`, 3.5M — the blueprint model.
///
/// Sized for its corpus rather than scaled down from `tiny`. The vocabulary is
/// 495 tokens instead of 32,768, so the embedding costs almost nothing and the
/// whole budget goes into the stack.
///
/// Attention is unwindowed here, which is the one place this disagrees with
/// every language preset. The task is placing entities that must not overlap
/// ones already placed, and a 64-entity blueprint is 400 tokens of history that
/// all of it depends on: a 128-token window would hide two thirds of the design
/// being built. Quadratic attention over 512 tokens at `d_model 192` is a
/// rounding error against the SSD scan.
pub fn nano() -> Model {
    Model::new(VOCAB, 192, 8)
        .with_seq_len(SEQ_LEN)
        .with_state_rank(32)
        .with_head_dim(32)
        .with_n_groups(1)
        .with_mimo_rank(1)
        .with_attn_period(Some(4))
        .with_attn_heads(6)
        .with_attn_kv_heads(2)
        .with_attn_window(None)
        .with_ffn_mult(2.0)
        .with_tied_embeddings(true)
}

/// Steps that cover [`Model::chinchilla_tokens`] at `tokens_per_step` a step.
///
/// The token batch is `micro_batch · accum · seq_len`; the caller passes it in
/// rather than having it guessed, because the batch is the memory decision and
/// this is only the arithmetic that follows from it.
pub fn chinchilla_steps(model: &Model, tokens_per_step: usize) -> usize {
    model.chinchilla_tokens().div_ceil(tokens_per_step.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_member_of_the_family_is_buildable() {
        for size in Size::ALL {
            size.model().validate().unwrap_or_else(|error| panic!("{}: {error}", size.name()));
        }
    }

    /// The corpus is built against this shape. A mismatch is caught by `train`
    /// as a vocabulary error, but only after the shards have been written.
    #[test]
    fn nano_matches_the_blueprint_corpus() {
        let cfg = nano();

        assert_eq!(cfg.vocab_size, VOCAB);
        assert!(cfg.seq_len >= SEQ_LEN);
        // Every entity already placed constrains the next one, so attention has
        // to reach back over a whole blueprint rather than a slice of one.
        assert_eq!(cfg.attn_window, None);
    }

    #[test]
    fn nano_stays_the_size_the_harness_reports() {
        let total = nano().budget().total;

        assert!((3.0e6..4.0e6).contains(&(total as f64)), "nano is {total}");
    }

    /// 20 tokens per parameter is about 70M tokens, which the measured 84k
    /// tokens/s on the target card covers in a quarter of an hour — the whole
    /// reason the rule is affordable here at all.
    #[test]
    fn the_chinchilla_budget_is_twenty_tokens_a_parameter() {
        let cfg = nano();
        let batch = 32 * cfg.seq_len;

        let steps = chinchilla_steps(&cfg, batch);

        assert!(steps * batch >= 20 * cfg.budget().total, "{steps} steps of {batch} tokens");
        assert!((60e6..80e6).contains(&(cfg.chinchilla_tokens() as f64)));
    }
}
