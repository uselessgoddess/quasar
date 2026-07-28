//! Reproduce the `tiny-turbo` divergence on a model small enough for a CPU.
//!
//! Issue #23 reports three failures of the same run — fp16 head+FFN+Mamba,
//! fp16 head only, and plain fp32 — all inside a hundred steps of
//! `--lr 3e-3 --warmup 20`. This probe keeps the whole recipe (the WSD
//! schedule, the Muon/AdamW split, gradient clipping, the dynamic loss scaler)
//! and shrinks only the shape, so the same question can be asked repeatedly
//! without a GPU: is the blow-up in the code or in the hyperparameters?
//!
//! ```text
//! cargo run --release --example divergence_probe -- --steps 200 --lr 3e-3 --warmup 20
//! ```

use std::path::{Path, PathBuf};

use anyhow::Result;
use burn::prelude::*;
use clap::Parser;
use quasar::config;
use quasar::data::shard;
use quasar::train::{self, Run};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Parser)]
#[command(about = "Run the production training loop on a CPU-sized hybrid")]
struct Args {
    /// Where to build the synthetic corpus and the checkpoints.
    #[arg(long)]
    work: Option<PathBuf>,
    #[arg(long, default_value_t = 512)]
    vocab: usize,
    #[arg(long, default_value_t = 128)]
    d_model: usize,
    #[arg(long, default_value_t = 6)]
    layers: usize,
    #[arg(long, default_value_t = 256)]
    seq: usize,
    #[arg(long, default_value_t = 200)]
    steps: usize,
    #[arg(long, default_value_t = 4)]
    micro_batch: usize,
    #[arg(long, default_value_t = 8)]
    accum: usize,
    #[arg(long, default_value_t = 3e-3)]
    lr: f64,
    #[arg(long, default_value_t = 20)]
    warmup: usize,
    #[arg(long, default_value_t = 1.0)]
    clip: f32,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    muon: bool,
    /// Tokens in the synthetic corpus.
    #[arg(long, default_value_t = 1 << 20)]
    tokens: usize,
    #[arg(long, default_value_t = 1337)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let work = match args.work {
        Some(work) => work,
        None => std::env::temp_dir().join("quasar-divergence-probe"),
    };
    let _ = std::fs::remove_dir_all(&work);

    corpus(&work.join("shards/train"), args.vocab, args.tokens, args.seed)?;
    corpus(&work.join("shards/valid"), args.vocab, args.tokens / 16, args.seed ^ 1)?;

    let heads = (args.d_model / 64).max(1);
    let cfg = config::Model::new(args.vocab, args.d_model, args.layers)
        .with_seq_len(args.seq)
        .with_state_rank(64)
        .with_head_dim(args.d_model / heads)
        .with_n_groups(1)
        .with_mimo_rank(1)
        .with_attn_period(Some(5))
        .with_attn_heads(heads)
        .with_attn_kv_heads(1)
        .with_attn_window(Some(64))
        .with_ffn_mult(2.0)
        .with_tied_embeddings(true);
    cfg.validate().expect("the probe shape is a valid model");
    println!("probe shape: {} params", cfg.budget().total);

    let run = Run::new()
        .with_steps(args.steps)
        .with_micro_batch(args.micro_batch)
        .with_accum(args.accum)
        .with_lr(args.lr)
        .with_warmup(args.warmup)
        .with_decay(args.steps / 4)
        .with_clip(args.clip)
        .with_muon(args.muon)
        .with_checkpointing(false)
        .with_log_every(1)
        .with_eval_every(0)
        .with_save_every(0)
        .with_seed(args.seed);

    let device = Device::default();
    let outcome = train::run(&cfg, &run, &work.join("shards"), &work.join("run"), &device);
    match outcome {
        Ok(()) => println!("probe finished {} steps without diverging", args.steps),
        Err(error) => println!("probe failed: {error:?}"),
    }
    Ok(())
}

/// A corpus a small model can actually fit: a deterministic second-order
/// recurrence over the vocabulary with a tenth of the tokens drawn at random,
/// which leaves a learnable signal and a floor the loss cannot pass.
fn corpus(dir: &Path, vocab: usize, tokens: usize, seed: u64) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut writer = shard::Writer::create(dir, vocab, 0)?;
    let (mut a, mut b) = (1u64, 2u64);
    let mut document = Vec::with_capacity(256);
    let mut written = 0;
    while written < tokens {
        document.clear();
        for _ in 0..256 {
            let next = if rng.random_bool(0.1) {
                rng.random_range(0..vocab as u64)
            } else {
                (a * 7 + b * 3 + 1) % vocab as u64
            };
            (a, b) = (b, next);
            document.push(next as u16);
        }
        writer.push(&document, document.len())?;
        written += document.len();
    }
    writer.finish()?;
    Ok(())
}
