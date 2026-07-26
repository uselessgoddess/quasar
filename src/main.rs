//! `quasar` — the command line around the crate.
//!
//! The pipeline is four commands in order: `budget` to see what a preset costs,
//! `tokenizer` to fit a vocabulary, `prepare` to turn a download into shards,
//! `train` to run. `eval` and `generate` inspect what came out.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Args, Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};

use quasar::data::{Batcher, Corpus, Shards, Tokenizer, prepare};
use quasar::model::Quasar;
use quasar::train::checkpoint;
use quasar::{config, eval, generate, train};

#[derive(Parser)]
#[command(name = "quasar", version, about = "A Mamba-3 language model on one GPU")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Parameters, FLOPs and memory of a preset.
    Budget {
        #[arg(value_enum, default_value_t = Preset::Tiny)]
        preset: Preset,
        /// Micro-batch the activation estimate is for.
        #[arg(long, default_value_t = 1)]
        micro_batch: usize,
        #[command(flatten)]
        shape: Shape,
    },
    /// Fit a byte-level BPE vocabulary on the corpus.
    Tokenizer {
        /// Files or directories of parquet/jsonl/txt documents.
        #[arg(required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long, default_value = "data/tokenizer.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 32_768)]
        vocab_size: usize,
        /// Documents to fit on. The full corpus is not needed and would take
        /// hours; BPE merges converge long before it.
        #[arg(long, default_value_t = 2_000_000)]
        docs: usize,
        #[arg(long, default_value = "text")]
        field: String,
    },
    /// Tokenize the corpus into `train/` and `valid/` shards.
    Prepare {
        #[arg(required = true)]
        corpus: Vec<PathBuf>,
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "data/shards")]
        out: PathBuf,
        #[arg(long, default_value = "text")]
        field: String,
    },
    /// Train, resuming from the newest checkpoint under `--out`.
    Train {
        #[arg(value_enum, default_value_t = Preset::Tiny)]
        preset: Preset,
        #[arg(long, default_value = "data/shards")]
        data: PathBuf,
        #[arg(long, default_value = "runs/tiny")]
        out: PathBuf,
        #[command(flatten)]
        run: Overrides,
        #[command(flatten)]
        shape: Shape,
    },
    /// Score a checkpoint on the validation shards.
    Eval {
        run: PathBuf,
        #[arg(long, default_value = "data/shards")]
        data: PathBuf,
        #[arg(long, default_value_t = 100)]
        batches: usize,
        #[arg(long, default_value_t = 8)]
        batch: usize,
    },
    /// Continue a prompt with a checkpoint.
    Generate {
        run: PathBuf,
        #[arg(long, default_value = "\n")]
        prompt: String,
        /// A file of prompts instead of one: jsonl with a `prompt` field, or
        /// one prompt per line. Every other field of a jsonl record is passed
        /// through to the output.
        #[arg(long, conflicts_with = "prompt")]
        prompts: Option<PathBuf>,
        /// Write jsonl here instead of printing. One record per prompt, with
        /// `text` holding prompt and continuation together.
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value = "data/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        #[arg(long, default_value_t = 1337)]
        seed: u64,
    },
}

/// The run knobs worth changing from the command line; the rest live in
/// `run.json` next to the checkpoints and are read back on resume.
#[derive(Args)]
struct Overrides {
    #[arg(long)]
    steps: Option<usize>,
    #[arg(long)]
    micro_batch: Option<usize>,
    #[arg(long)]
    accum: Option<usize>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    warmup: Option<usize>,
    #[arg(long)]
    decay: Option<usize>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    save_every: Option<usize>,
    #[arg(long)]
    eval_every: Option<usize>,
    /// Steps between the loss/throughput lines on stdout. Those lines are what
    /// the metric plots are drawn from, so a short run wants a short interval.
    #[arg(long)]
    log_every: Option<usize>,
    /// Muon on the hidden matrices; `false` puts everything on AdamW.
    #[arg(long)]
    muon: Option<bool>,
    /// Recompute activations in the backward.
    #[arg(long)]
    checkpointing: Option<bool>,
    /// SSD algorithm: serial retains intermediates for speed; recalculated saves memory.
    #[arg(long, value_enum)]
    ssd: Option<Ssd>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Ssd {
    Minimal,
    Serial,
    Recalculated,
}

/// The model-shape knobs worth sweeping without editing a preset, because they
/// are the ones that decide whether a micro-batch fits. `quasar budget` answers
/// that before a run allocates anything.
#[derive(Args, Clone)]
struct Shape {
    #[arg(long)]
    seq_len: Option<usize>,
    #[arg(long)]
    state_rank: Option<usize>,
    #[arg(long)]
    mimo_rank: Option<usize>,
    #[arg(long)]
    expand: Option<usize>,
    /// Sliding-window radius; `0` means full causal attention.
    #[arg(long)]
    attn_window: Option<usize>,
    /// Attention every n-th layer; `0` means a pure-SSM stack.
    #[arg(long)]
    attn_period: Option<usize>,
    /// SSD scan chunk length; unset keeps the largest divisor of `seq_len`
    /// below burn-mamba's own rule of thumb.
    #[arg(long)]
    ssd_chunk: Option<usize>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Preset {
    Tiny,
    /// `tiny`, cut down to what trains fastest inside 16 GB.
    TinyTurbo,
    Base,
    /// 3.5M, 657-token vocabulary — the Factorio blueprint family.
    FactorioNano,
    Toy,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Budget { preset, micro_batch, shape } => {
            budget(shape.apply(preset.config()), micro_batch)
        }
        Command::Tokenizer { corpus, out, vocab_size, docs, field } => {
            tokenizer(&corpus, &out, vocab_size, docs, &field)
        }
        Command::Prepare { corpus, tokenizer, out, field } => {
            let corpus = Corpus::open(&corpus, &field)?;
            let tokenizer = Tokenizer::load(&tokenizer)?;
            let prepared = prepare::run(&corpus, &tokenizer, &out)?;
            println!("train {} tokens", prepared.train.tokens);
            println!("valid {} tokens", prepared.valid.tokens);
            Ok(())
        }
        Command::Train { preset, data, out, run, shape } => {
            // The tokenizer decides the vocabulary, not the preset: a corpus
            // whose BPE stopped short of the requested merges is common, and
            // silently building a wider embedding would be dead parameters.
            let mut cfg = shape.apply(preset.config());
            cfg.vocab_size = Shards::open(&data.join("train"))?.meta().vocab_size;
            let run = run.apply(preset.run_defaults());
            train::run(&cfg, &run, &data, &out, &Device::default())?;
            Ok(())
        }
        Command::Eval { run, data, batches, batch } => evaluate(&run, &data, batches, batch),
        Command::Generate {
            run,
            prompt,
            prompts,
            out,
            tokenizer,
            tokens,
            temperature,
            top_k,
            seed,
        } => {
            let sampler = generate::Sampler { temperature, top_k, max_tokens: tokens, seed };
            let asked = match prompts {
                Some(path) => asks(&path)?,
                None => vec![Ask { prompt, record: serde_json::Map::new() }],
            };
            sample(&run, &tokenizer, &asked, out.as_deref(), &sampler)
        }
    }
}

/// What a preset costs, in the currencies that decide whether it fits.
fn budget(cfg: config::Model, micro_batch: usize) -> Result<()> {
    cfg.validate()?;
    let budget = cfg.budget();
    let params = budget.total as f64;
    let gib = |bytes_per_param: f64| params * bytes_per_param / (1 << 30) as f64;
    let flops = cfg.flops_per_token();

    println!("{budget}\n");
    println!("seq_len          {}", cfg.seq_len);
    println!("ssd chunk        {}", cfg.ssd_chunk_len());
    println!("attn keys/query  {}", cfg.attn_pairs() / cfg.seq_len);
    println!("fwd FLOPs/token  {:.1}M", flops / 1e6);
    println!("step FLOPs/token {:.1}M", 3.0 * flops / 1e6);
    // What a finished run of this model costs, before any wall clock is
    // involved: 20 tokens per parameter, and the steps that buys at the batch
    // being asked about.
    let batch = micro_batch * cfg.seq_len;
    println!("chinchilla       {:.1}M tokens", cfg.chinchilla_tokens() as f64 / 1e6);
    println!("  at batch {batch:<6} {} steps", cfg.chinchilla_tokens().div_ceil(batch));
    // Weights, gradients and two Adam moments. Pure bf16 is 8 B/param; keeping
    // fp32 master weights and moments doubles it, which is what decides whether
    // a preset fits 16 GB before a single activation is allocated.
    println!("states bf16      {:.2} GiB", gib(8.0));
    println!("states fp32      {:.2} GiB", gib(16.0));
    // Muon keeps one momentum buffer where AdamW keeps two moments, and every
    // matrix in the stack is on Muon — only the vocabulary and the norms are not.
    println!("states muon      {:.2} GiB", gib(12.0));

    // States are the easy half: they are the same every step. The activations
    // are what an OOM at `micro_batch 2` is actually about, so they get their own
    // breakdown and the micro-batch the card has room for.
    let sixteen = (16u64 << 30) as f64;
    println!("\nactivations at micro_batch {micro_batch} (fp32, estimated)");
    println!("{}", cfg.activations(micro_batch));
    println!("\nmicro_batch in 16 GiB, muon states {}", cfg.micro_batch_within(sixteen, 12.0));
    println!("micro_batch in 16 GiB, fp32 adamw  {}", cfg.micro_batch_within(sixteen, 16.0));
    Ok(())
}

fn tokenizer(
    corpus: &[PathBuf],
    out: &Path,
    vocab_size: usize,
    docs: usize,
    field: &str,
) -> Result<()> {
    let corpus = Corpus::open(corpus, field)?;
    println!("fitting {vocab_size} tokens on up to {docs} documents");

    let stream = corpus.docs().filter_map(Result::ok).take(docs);
    let tokenizer = Tokenizer::train(stream, vocab_size)?;

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    tokenizer.save(out)?;
    println!("wrote {} tokens to {}", tokenizer.vocab_size(), out.display());
    Ok(())
}

fn evaluate(run: &Path, data: &Path, batches: usize, batch: usize) -> Result<()> {
    let (cfg, dir) = trained(run)?;
    let device = Device::default();
    let mut model = Quasar::new(&cfg, &device);
    checkpoint::weights(&dir, &mut model)?;

    let shards = Shards::open(&data.join("valid"))?;
    let valid = Batcher::new(shards, cfg.seq_len, batch, 0);
    println!("{}", eval::evaluate(&model, &valid, batches, &device));
    Ok(())
}

/// A prompt to continue, and whatever else the file said about it.
struct Ask {
    prompt: String,
    /// The input record, passed through to the output so that whoever wrote the
    /// prompts gets their own metadata back next to the generation.
    record: serde_json::Map<String, serde_json::Value>,
}

/// Continue every prompt in `asked` with the newest checkpoint of `run`.
///
/// Loading a checkpoint costs far more than sampling from it, so a sweep of two
/// hundred prompts belongs in one process rather than two hundred. Each prompt
/// samples from its own seed: sharing one would draw the same noise every time
/// and pass it off as a varied sweep.
fn sample(
    run: &Path,
    tokenizer: &Path,
    asked: &[Ask],
    out: Option<&Path>,
    sampler: &generate::Sampler,
) -> Result<()> {
    let (cfg, dir) = trained(run)?;
    let device = Device::default();
    let mut model = Quasar::new(&cfg, &device);
    checkpoint::weights(&dir, &mut model)?;
    let tokenizer = Tokenizer::load(tokenizer)?;

    let bar = ProgressBar::new(asked.len() as u64).with_style(
        ProgressStyle::with_template("{bar:32} {pos}/{len} sampled in {elapsed}").unwrap(),
    );
    if out.is_none() {
        bar.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }

    let mut lines = String::new();
    for (index, ask) in asked.iter().enumerate() {
        let seeded =
            generate::Sampler { seed: sampler.seed.wrapping_add(index as u64), ..*sampler };
        let text =
            generate::generate(&model, &tokenizer, &ask.prompt, cfg.seq_len, &seeded, &device)?;
        match out {
            // `text` already holds the prompt as well as the continuation,
            // which is what a grader needs: half a blueprint parses as none.
            Some(_) => {
                let mut record = ask.record.clone();
                record.insert("prompt".into(), ask.prompt.clone().into());
                record.insert("text".into(), text.into());
                lines.push_str(&serde_json::to_string(&record)?);
                lines.push('\n');
            }
            None => println!("{text}"),
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    if let Some(path) = out {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, lines).with_context(|| format!("cannot write {}", path.display()))?;
        println!("{} samples -> {}", asked.len(), path.display());
    }
    Ok(())
}

/// Prompts from a jsonl file, or from a plain file of one prompt per line.
///
/// Both spellings because both get written: the harness emits jsonl with the
/// spec it drew the prompt from, and a person testing a checkpoint writes a
/// text file.
fn asks(path: &Path) -> Result<Vec<Ask>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    let mut asked = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        let at = || format!("{}:{}", path.display(), index + 1);
        if line.is_empty() {
            continue;
        }
        if !line.starts_with('{') {
            asked.push(Ask { prompt: line.to_owned(), record: serde_json::Map::new() });
            continue;
        }
        let record: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(line).with_context(|| format!("{} is not a json object", at()))?;
        let prompt = record
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .with_context(|| format!("{} has no `prompt` string", at()))?
            .to_owned();
        asked.push(Ask { prompt, record });
    }
    if asked.is_empty() {
        anyhow::bail!("no prompts in {}", path.display());
    }
    Ok(asked)
}

/// The config and a checkpoint of a run.
///
/// `run` is either the run directory, which resolves to its newest checkpoint,
/// or one checkpoint inside it, which is taken as given. The second form is
/// what scoring a run over time needs: the interesting question is not only
/// what the finished model builds but what it was building at step 200, and
/// that is a directory the run already wrote.
fn trained(run: &Path) -> Result<(config::Model, PathBuf)> {
    let (root, dir) = match run.join("state.json").is_file() {
        true => (run.parent().unwrap_or(run).to_path_buf(), run.to_path_buf()),
        false => (
            run.to_path_buf(),
            checkpoint::latest(run)
                .with_context(|| format!("no checkpoint in {}", run.display()))?,
        ),
    };
    let cfg = config::Model::load(root.join("model.json"))
        .with_context(|| format!("no model.json in {}", root.display()))?;
    println!("{}", dir.display());
    Ok((cfg, dir))
}

impl Preset {
    fn config(self) -> config::Model {
        match self {
            Self::Tiny => config::Model::tiny(),
            Self::TinyTurbo => config::Model::tiny_turbo(),
            Self::Base => config::Model::base(),
            Self::FactorioNano => config::factorio::nano(),
            Self::Toy => config::Model::toy(),
        }
    }

    /// Run defaults that have been validated for one exact production shape.
    ///
    /// The retained graph fits `tiny-turbo` at micro-batch 4 without outer
    /// checkpointing on the 16-GB target card. Accumulation rises inversely so
    /// the default token budget is unchanged. Larger presets keep the
    /// memory-saving defaults; callers can still override every path.
    ///
    /// `factorio-nano` is the other end: at 3.5M parameters the memory-saving
    /// defaults buy nothing and cost a third of the throughput, so both come
    /// off and the batch goes into micro-batch where it runs as one pass.
    ///
    /// Its length is not a wall-clock choice. The Chinchilla ratio asks for 20
    /// tokens per parameter, which at a 16k-token batch is about 4,300 steps —
    /// a quarter of an hour at the throughput this preset measured. A shorter
    /// run is not a smaller experiment, it is one whose curves have not
    /// separated the model from its initialisation yet.
    fn run_defaults(self) -> train::Run {
        match self {
            Self::TinyTurbo => train::Run::new()
                .with_micro_batch(4)
                .with_accum(32)
                .with_checkpointing(false)
                .with_ssd_mode(Some(config::SsdMode::Serial)),
            Self::FactorioNano => {
                let cfg = config::factorio::nano();
                let (micro, accum) = (32, 1);
                let steps = config::factorio::chinchilla_steps(&cfg, micro * accum * cfg.seq_len);
                train::Run::new()
                    .with_steps(steps)
                    .with_micro_batch(micro)
                    .with_accum(accum)
                    // Half the shared default, and measured rather than chosen.
                    // At 3e-3 this preset sits on the edge of stability: of six
                    // runs on the maintainer's card, the two short ones that
                    // finished did so with the loss already oscillating, and
                    // four came apart -- at steps 415, ~600, ~530 and 1704 --
                    // each with the same signature, a loss that stops falling,
                    // swings wider over a few hundred steps, and then leaves
                    // the format. Two of those were the same commit and the
                    // same corpus as a run that survived, so it is not a
                    // recipe that fails, it is one that fails half the time,
                    // and a run twenty times longer than the ones that
                    // survived does not get to be a coin flip.
                    .with_lr(1.5e-3)
                    // The preset's own proportions: 5% warmup, 20% decay.
                    .with_warmup(steps / 20)
                    .with_decay(steps / 5)
                    .with_eval_every(steps / 20)
                    .with_save_every(steps / 4)
                    .with_log_every(steps / 60)
                    .with_checkpointing(false)
                    .with_ssd_mode(Some(config::SsdMode::Serial))
            }
            Self::Tiny | Self::Base | Self::Toy => train::Run::new(),
        }
    }
}

impl Shape {
    fn apply(&self, mut cfg: config::Model) -> config::Model {
        macro_rules! set {
            ($($field:ident),*) => {$(if let Some(value) = self.$field { cfg.$field = value; })*};
        }
        set!(seq_len, state_rank, mimo_rank, expand);
        // `0` is how a flag says "none" — clap has no `--attn-window=` spelling
        // for `Option<usize>` inside an `Option`.
        if let Some(window) = self.attn_window {
            cfg.attn_window = (window > 0).then_some(window);
        }
        if let Some(period) = self.attn_period {
            cfg.attn_period = (period > 0).then_some(period);
        }
        if let Some(chunk) = self.ssd_chunk {
            cfg.ssd_chunk = Some(chunk);
        }
        cfg
    }
}

impl Overrides {
    fn apply(&self, mut run: train::Run) -> train::Run {
        macro_rules! set {
            ($($field:ident),*) => {$(if let Some(value) = self.$field { run.$field = value; })*};
        }
        set!(
            steps,
            micro_batch,
            accum,
            lr,
            warmup,
            decay,
            seed,
            save_every,
            eval_every,
            log_every,
            muon,
            checkpointing
        );
        if let Some(ssd) = self.ssd {
            run.ssd_mode = Some(match ssd {
                Ssd::Minimal => config::SsdMode::Minimal,
                Ssd::Serial => config::SsdMode::Serial,
                Ssd::Recalculated => config::SsdMode::Recalculated,
            });
        }
        run
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_measured_presets_leave_the_training_defaults() {
        let turbo = Preset::TinyTurbo.run_defaults();

        assert_eq!(turbo.ssd_mode, Some(config::SsdMode::Serial));
        assert_eq!((turbo.micro_batch, turbo.accum), (4, 32));
        assert!(!turbo.checkpointing);

        let nano = Preset::FactorioNano.run_defaults();

        assert_eq!(nano.ssd_mode, Some(config::SsdMode::Serial));
        assert_eq!((nano.micro_batch, nano.accum), (32, 1));
        assert!(!nano.checkpointing);
        // The schedule has to fit inside the run it is scheduling.
        assert!(nano.warmup + nano.decay < nano.steps);
        // And it runs under the shared peak rate, which this preset diverged at.
        assert!(nano.lr < train::Run::new().lr, "{} is back on the edge", nano.lr);
        // And the run has to be long enough to be worth reading: 20 tokens per
        // parameter, the floor `config::factorio` derives everything from.
        let cfg = Preset::FactorioNano.config();
        let seen = nano.steps * nano.micro_batch * nano.accum * cfg.seq_len;
        assert!(seen >= cfg.chinchilla_tokens(), "{seen} tokens is short");

        for preset in [Preset::Tiny, Preset::Base, Preset::Toy] {
            let run = preset.run_defaults();
            assert_eq!(run.ssd_mode, None);
            assert_eq!((run.micro_batch, run.accum), (8, 16));
            assert!(run.checkpointing);
        }
    }

    #[test]
    fn prompts_are_read_from_jsonl_or_from_plain_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        fs::write(
            &path,
            "{\"prompt\": \"<bp> <spec>\", \"spec\": {\"kind\": \"belt-lane\"}}\n\n<bp> plain\n",
        )
        .unwrap();

        let asked = asks(&path).unwrap();

        assert_eq!(asked.len(), 2, "the blank line is not a prompt");
        assert_eq!(asked[0].prompt, "<bp> <spec>");
        // Carried through, so a grader keeps the spec that produced the prompt.
        assert!(asked[0].record.contains_key("spec"));
        assert_eq!(asked[1].prompt, "<bp> plain");
        assert!(asked[1].record.is_empty());
    }

    #[test]
    fn a_prompt_file_that_says_nothing_is_an_error_not_an_empty_sweep() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prompts.jsonl");
        fs::write(&path, "\n \n").unwrap();

        assert!(asks(&path).is_err());
    }

    #[test]
    fn a_run_resolves_to_its_newest_checkpoint_and_a_checkpoint_to_itself() {
        let dir = tempfile::tempdir().unwrap();
        let run = dir.path();
        config::Model::toy().save(run.join("model.json")).unwrap();
        for step in [10usize, 200] {
            let at = checkpoint::dir(run, step);
            fs::create_dir_all(&at).unwrap();
            fs::write(at.join("state.json"), "{\"step\": 0, \"tokens\": 0}").unwrap();
        }

        let (_, newest) = trained(run).unwrap();
        let (_, asked) = trained(&checkpoint::dir(run, 10)).unwrap();

        assert_eq!(newest, checkpoint::dir(run, 200));
        // Scoring a run over time depends on this: `model.json` lives in the
        // run directory, not in the checkpoint that was named.
        assert_eq!(asked, checkpoint::dir(run, 10));
    }

    /// Every member of the Factorio family is reachable from the CLI. The
    /// family is where the shapes are checked against the corpus; this only
    /// pins that adding one and forgetting the `--preset` name is a test
    /// failure rather than a model nobody can train.
    #[test]
    fn the_factorio_family_is_spelled_out_in_the_preset_list() {
        let names: Vec<_> = Preset::value_variants()
            .iter()
            .filter_map(|preset| preset.to_possible_value())
            .map(|value| value.get_name().to_string())
            .collect();

        for size in config::factorio::Size::ALL {
            assert!(names.contains(&size.name().to_string()), "{} is not a preset", size.name());
        }
        assert_eq!(Preset::FactorioNano.config(), config::factorio::Size::Nano.model());
    }
}
