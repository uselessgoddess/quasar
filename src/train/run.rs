//! The training loop.
//!
//! Everything the loop needs is a pure function of the step index — the batch,
//! the learning rate, the schedule — so a crashed run resumes into exactly the
//! stream it left. The loop itself stays a plain `for`: Burn's official metric
//! renderer is driven from [`super::ui`] without handing it the custom Muon /
//! AdamW optimizer split or the deterministic checkpoint lifecycle.

use std::io;
use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::prelude::*;
use burn::store::ModuleSnapshot;
use burn::tensor::FloatDType;

use crate::data::{Batcher, Shards};
use crate::model::Quasar;
use crate::train::checkpoint::{self, State};
use crate::train::optim::{DynamicLossScaler, Optim};
use crate::train::schedule::Wsd;
use crate::train::ui::Dashboard;
use crate::{config, eval};

/// Everything about a run that is not the model itself.
///
/// The defaults are the compute-efficient `tiny` recipe: about 20 training
/// tokens per parameter. Wall-clock time depends on measured device throughput.
#[derive(Config, Debug)]
pub struct Run {
    #[config(default = 12_500)]
    pub steps: usize,
    /// Sequences per forward pass. This is the VRAM knob — raise it until the
    /// card refuses, then compensate with `accum`.
    #[config(default = 8)]
    pub micro_batch: usize,
    /// Forward/backward passes per optimizer step. `micro_batch * accum *
    /// seq_len` is the token batch, which is what the learning rate is tuned
    /// against, so the two must move together.
    #[config(default = 16)]
    pub accum: usize,
    #[config(default = 3e-3)]
    pub lr: f64,
    /// Final rate as a fraction of `lr`.
    #[config(default = 0.1)]
    pub lr_floor: f64,
    #[config(default = 400)]
    pub warmup: usize,
    #[config(default = 2_500)]
    pub decay: usize,
    #[config(default = 0.1)]
    pub weight_decay: f32,
    #[config(default = 1.0)]
    pub clip: f32,
    /// Orthogonalise the update of the hidden matrices instead of adapting it
    /// per coordinate. One momentum buffer rather than two moments, which is
    /// 4 B/param less state — see [`crate::train::Optim`].
    #[config(default = true)]
    pub muon: bool,
    /// Recompute activations in the backward instead of keeping them. Trades
    /// roughly a third more compute for the activation memory that decides
    /// whether `base` fits at all.
    #[config(default = true)]
    pub checkpointing: bool,
    /// SSD-specific memory/speed tradeoff. `None` keeps burn-mamba's
    /// memory-efficient recalculated backward; `Serial` retains intermediates
    /// and is faster when they fit. Optional so older run files remain readable.
    #[config(default = "None")]
    pub ssd_mode: Option<config::SsdMode>,
    /// Reduced compute dtype for the output-head GEMM only. Parameters,
    /// optimizer state, hidden states, logits, softmax and loss remain fp32.
    /// Optional so older run files and unmeasured presets remain fp32.
    #[config(default = "None")]
    pub head_dtype: Option<config::HeadDtype>,
    /// Reduced compute dtype for the three FFN GEMMs in every block. Master
    /// weights, norms, SwiGLU elementwise math and residuals stay fp32.
    #[config(default = "None")]
    pub ffn_dtype: Option<config::FfnDtype>,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 20)]
    pub log_every: usize,
    #[config(default = 1_000)]
    pub eval_every: usize,
    #[config(default = 20)]
    pub eval_batches: usize,
    #[config(default = 2_000)]
    pub save_every: usize,
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Checkpoint(checkpoint::Error),
    /// The shards were tokenized with a different vocabulary than the model
    /// was built for, which shows up as an out-of-bounds gather rather than
    /// anything readable if it is allowed through.
    Vocab {
        config: usize,
        shards: usize,
    },
    /// The loss stopped being a number.
    ///
    /// A diverged run is not a worse run, it is not a run: one NaN reaches
    /// every weight within a step, every sample afterwards is empty, and the
    /// grade sheet at the end reports a flat zero that reads exactly like a
    /// model which learnt nothing. That is how a four-thousand-step run
    /// finished green with a dead model in it. The loss is already read on
    /// logging steps, so noticing costs nothing until it fires.
    Diverged {
        step: usize,
        params: Vec<String>,
    },
    /// Even the minimum fp16 loss scale produced a non-finite gradient. The
    /// rejected update never reached either optimizer or the model.
    NonfiniteGradient {
        step: usize,
        loss_scale: f64,
    },
}

/// Train `cfg` on the shards under `data`, checkpointing into `out`.
///
/// Resumes by itself from the newest checkpoint in `out`, because that is what
/// a run interrupted at 3am needs to do when it is restarted at 9.
pub fn run(
    cfg: &config::Model,
    run: &Run,
    data: &Path,
    out: &Path,
    device: &Device,
) -> Result<(), Error> {
    let inference = device.clone();
    let device = device.clone().autodiff();
    // Checkpointing is a property of the autodiff device: the graph keeps the
    // inputs of a checkpointed span and replays it in the backward.
    let device = if run.checkpointing { device.gradient_checkpointing() } else { device };
    device.seed(run.seed);

    let train = batcher(&data.join("train"), cfg.seq_len, run)?;
    let valid = batcher(&data.join("valid"), cfg.seq_len, run)?;

    let shards = train.shards().meta().vocab_size;
    if shards != cfg.vocab_size {
        return Err(Error::Vocab { config: cfg.vocab_size, shards });
    }

    let head_dtype = run.head_dtype.as_ref().map(|dtype| match dtype {
        config::HeadDtype::F16 => FloatDType::F16,
    });
    let ffn_dtype = run.ffn_dtype.as_ref().map(|dtype| match dtype {
        config::FfnDtype::F16 => FloatDType::F16,
    });
    let mut model = Quasar::new_with_ssd_and_dtypes(
        cfg,
        run.ssd_mode.clone().unwrap_or_default(),
        head_dtype,
        ffn_dtype,
        &device,
    );
    let mut optim = Optim::new(run, &model);

    std::fs::create_dir_all(out).map_err(Error::Io)?;
    cfg.save(out.join("model.json")).map_err(Error::Io)?;
    run.save(out.join("run.json")).map_err(Error::Io)?;

    let mut state = State { step: 0, tokens: 0, loss_scaler: None };
    if let Some(dir) = checkpoint::latest(out) {
        let (loaded, resumed) = checkpoint::load(&dir, &mut model, optim)?;
        (optim, state) = (loaded, resumed);
        println!("resuming {} at step {}", dir.display(), state.step);
    }
    let mut loss_scaler = (run.head_dtype.is_some() || run.ffn_dtype.is_some())
        .then(|| state.loss_scaler.unwrap_or_else(|| DynamicLossScaler::new(200)));
    state.loss_scaler = loss_scaler;

    let schedule = Wsd::new(run.lr, run.lr_floor, run.warmup, run.decay, run.steps);
    let per_step = (run.micro_batch * run.accum * cfg.seq_len) as u64;
    let total_tokens = run.steps as u64 * per_step;
    let mut window = Window::new();
    println!(
        "training plan: {} optimizer steps | {:.2}B tokens | {} tokens/step",
        run.steps,
        total_tokens as f64 / 1e9,
        per_step
    );
    if let Some(scaler) = loss_scaler {
        let seams = match (run.head_dtype.is_some(), run.ffn_dtype.is_some()) {
            (true, true) => "output head + FFN projections",
            (true, false) => "output head",
            (false, true) => "FFN projections",
            (false, false) => unreachable!("a scaler requires an fp16 seam"),
        };
        println!(
            "training precision: fp16 {seams} | fp32 master/norm/residual/logits/loss | loss scale {:.0}",
            scaler.scale()
        );
    }
    let mut dashboard = Dashboard::new(run.steps, state.step);
    let tracing = trace_every();

    #[cfg(feature = "gpu")]
    if !dashboard.active() {
        println!(
            "warming up GPU kernels; Burn compiles, fuses and autotunes the first step before steady training"
        );
    }

    for step in state.step..run.steps {
        let lr = schedule.lr(step);
        // Reading a loss scalar blocks until the device catches up, so it is
        // read on logging steps only; the rest never leave the queue.
        let logging = due(step, run.log_every);
        let logged_loss = loop {
            let loss_scale = loss_scaler.as_ref().map_or(1.0, DynamicLossScaler::scale);
            let mut logged_loss = None;

            for micro in 0..run.accum {
                let batch = train.train((step * run.accum + micro) as u64, &device);
                let loss = model.loss(batch.input, batch.target);
                if logging {
                    // Keep the reduction on the device. Reading every micro-batch
                    // separately serialises the GPU queue `accum` times (96 times
                    // in the report that prompted issue #7).
                    let nll = loss.nll.clone().detach();
                    logged_loss = Some(match logged_loss.take() {
                        Some(total) => total + nll,
                        None => nll,
                    });
                }
                // Scaling here rather than after accumulating keeps the gradient
                // of an accumulated step identical to that of one big batch.
                let total = loss.total.div_scalar(run.accum as f64);
                let total =
                    if loss_scaler.is_some() { total.mul_scalar(loss_scale) } else { total };
                optim.accumulate(&model, total.backward());
            }

            let Some(scaler) = loss_scaler.as_mut() else {
                model = optim.step(lr, model);
                break logged_loss;
            };
            let (next, finite) = optim.step_scaled(lr, model, loss_scale);
            model = next;
            scaler.update(finite);
            if finite {
                break logged_loss;
            }
            if loss_scale <= DynamicLossScaler::MIN {
                return Err(Error::NonfiniteGradient { step: step + 1, loss_scale });
            }
            println!(
                "non-finite gradient at step {}; discarded update and retrying at loss scale {:.0}",
                step + 1,
                scaler.scale()
            );
        };
        state = State { step: step + 1, tokens: state.tokens + per_step, loss_scaler };
        window.steps += 1;

        if logging {
            window.loss = logged_loss
                .expect("a logging step contains at least one micro-batch")
                .div_scalar(run.accum as f64)
                .into_scalar::<f32>() as f64;
            if !window.loss.is_finite() {
                return Err(Error::Diverged { step: state.step, params: nonfinite(&model) });
            }
            let report = window.report(state, run, lr, per_step, cfg.flops_per_token());
            dashboard.train(
                state.step,
                report.loss,
                report.lr,
                report.throughput,
                report.tokens,
                report.eta_hours,
                report.tflops,
            );
            if !dashboard.active() {
                println!("{report}");
            }
            window = Window::new();
        }
        // Its own cadence rather than the logging one: a divergence is bounded
        // by two readings, and `log_every` is a sixtieth of the run — seventy
        // steps in which a scale can leave the format entirely.
        if due(step, tracing) {
            println!("  norms: step {} | {}", state.step, norms(&model));
        }
        if due(step, run.eval_every) {
            let report = eval::evaluate(&model.valid(), &valid, run.eval_batches, &inference);
            dashboard.valid(report);
            if !dashboard.active() {
                // The step is on the line because the line is plotted: read
                // back later, a validation point that has to be attributed to
                // whichever training line came before it is a point that lands
                // at zero whenever training logged nothing.
                println!("  valid: step {} | {report}", state.step);
            }
        }
        if due(step, run.save_every) {
            checkpoint::save(&checkpoint::dir(out, state.step), state, &model, &optim)?;
        }
        if dashboard.should_stop() {
            break;
        }
    }

    checkpoint::save(&checkpoint::dir(out, state.step), state, &model, &optim)?;
    let final_report = eval::evaluate(&model.valid(), &valid, run.eval_batches, &inference);
    dashboard.valid(final_report);
    dashboard.finish();
    println!("final: step {} | {final_report}", state.step);
    Ok(())
}

/// Every parameter of `model` that holds something which is not a number.
///
/// Reading the weights back is a device sync and a full copy, which is why this
/// runs once, on the way out of a run that has already failed.
fn nonfinite(model: &Quasar) -> Vec<String> {
    model
        .collect(None, None, false)
        .iter()
        .filter(|tensor| match tensor.to_data() {
            Ok(data) => data.iter::<f32>().any(|value| !value.is_finite()),
            // A tensor that cannot be read back cannot be cleared either, and
            // the run is being abandoned regardless.
            Err(_) => true,
        })
        .map(|tensor| tensor.full_path())
        .collect()
}

/// The four parameters with the largest RMS, largest first.
///
/// Divergence has a shape before it has a NaN: one tensor's scale climbs for a
/// few hundred steps and then leaves the range of the format. Printing the top
/// of that list as the run goes is what tells the difference between the
/// optimizer and the objective afterwards.
fn norms(model: &Quasar) -> String {
    let mut scales: Vec<(String, f32)> = model
        .collect(None, None, false)
        .iter()
        .filter_map(|tensor| {
            let data = tensor.to_data().ok()?;
            let (count, total) = data
                .iter::<f32>()
                .fold((0usize, 0f64), |(n, sum), v| (n + 1, sum + (v as f64) * (v as f64)));
            let rms = (total / count.max(1) as f64).sqrt() as f32;
            Some((tensor.full_path(), rms))
        })
        .collect();
    scales.sort_by(|a, b| b.1.total_cmp(&a.1));
    scales
        .iter()
        .take(4)
        .map(|(path, rms)| format!("{path} {rms:.3}"))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// How often to print [`norms`], in steps. Zero — the default — is never.
///
/// `QUASAR_TRACE_NORMS=10` traces every tenth step. It lives in the environment
/// rather than the run file because it is not part of the experiment: it is a
/// device sync and a full parameter read per traced step, which is exactly what
/// the loop is written to avoid, and it is wanted only after a run has already
/// misbehaved once. Anything unparseable reads as `1`, so a bare `=yes` traces
/// every step rather than silently doing nothing.
fn trace_every() -> usize {
    match std::env::var("QUASAR_TRACE_NORMS") {
        Err(_) => 0,
        Ok(value) => value.trim().parse().unwrap_or(1),
    }
}

fn batcher(dir: &Path, seq_len: usize, run: &Run) -> Result<Batcher, Error> {
    let shards = Shards::open(dir).map_err(Error::Io)?;
    Ok(Batcher::new(shards, seq_len, run.micro_batch, run.seed))
}

/// Whether `step` is the last of an `every`-sized period; `every == 0` disables.
fn due(step: usize, every: usize) -> bool {
    every > 0 && (step + 1).is_multiple_of(every)
}

/// What one log line reports: the loss of the step that was measured, and the
/// time since the previous line. The steps in between never leave the device.
struct Window {
    loss: f64,
    steps: usize,
    started: Instant,
}

impl Window {
    fn new() -> Self {
        Self { loss: 0.0, steps: 0, started: Instant::now() }
    }

    fn report(
        &self,
        state: State,
        run: &Run,
        lr: f64,
        per_step: u64,
        forward_flops_per_token: f64,
    ) -> TrainingReport {
        let seconds = self.started.elapsed().as_secs_f64();
        let throughput = (self.steps as u64 * per_step) as f64 / seconds;
        let (eta_hours, tflops) =
            speed(state.step, run.steps, per_step, throughput, forward_flops_per_token);
        TrainingReport {
            step: state.step,
            steps: run.steps,
            loss: self.loss,
            lr,
            throughput,
            tokens: state.tokens,
            eta_hours,
            tflops,
        }
    }
}

/// Convert the measured token rate into the two quantities needed to decide
/// whether a recipe is practical. Backward is conventionally 2× forward.
fn speed(
    step: usize,
    steps: usize,
    tokens_per_step: u64,
    throughput: f64,
    forward_flops_per_token: f64,
) -> (f64, f64) {
    let remaining_tokens = steps.saturating_sub(step) as f64 * tokens_per_step as f64;
    let eta_hours = remaining_tokens / throughput / 3_600.0;
    let tflops = throughput * 3.0 * forward_flops_per_token / 1e12;
    (eta_hours, tflops)
}

struct TrainingReport {
    step: usize,
    steps: usize,
    loss: f64,
    lr: f64,
    throughput: f64,
    tokens: u64,
    eta_hours: f64,
    tflops: f64,
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "optimizer step {}/{} | loss {loss:.4} | lr {lr:.2e} | {:.0} tok/s | \
             {:.2} TFLOP/s | {:.2}B tokens | ETA {eta}",
            self.step,
            self.steps,
            self.throughput,
            self.tflops,
            self.tokens as f64 / 1e9,
            loss = self.loss,
            lr = self.lr,
            eta = if self.eta_hours < 24.0 {
                format!("{:.1}h", self.eta_hours)
            } else {
                format!("{:.1}d", self.eta_hours / 24.0)
            },
        )
    }
}

impl From<checkpoint::Error> for Error {
    fn from(error: checkpoint::Error) -> Self {
        Self::Checkpoint(error)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Checkpoint(error) => write!(f, "{error}"),
            Self::Vocab { config, shards } => {
                write!(f, "model vocabulary {config} does not match the shards' {shards}")
            }
            Self::Diverged { step, params } => {
                writeln!(f, "training diverged at step {step}: the loss is not a number.")?;
                match params.split_first() {
                    None => write!(
                        f,
                        "every parameter is still finite, so the loss came apart in the forward"
                    )?,
                    Some((first, rest)) => {
                        // Module order, not the order they came apart in: by the
                        // time a logging step reads the loss the damage has
                        // spread, and what is worth printing is how far.
                        write!(f, "{} tensors are not finite, from {first}", params.len())?;
                        if !rest.is_empty() {
                            write!(f, " through {}", rest[rest.len() - 1])?;
                        }
                    }
                }
                write!(
                    f,
                    "\nlower --lr, lengthen --warmup, or take the hidden matrices off Muon with --muon false"
                )
            }
            Self::NonfiniteGradient { step, loss_scale } => write!(
                f,
                "training produced non-finite gradients at step {step} with minimum loss scale \
                 {loss_scale}; the update was discarded"
            ),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::shard;

    fn shards(dir: &Path) {
        let mut writer = shard::Writer::create(dir, 64, 0).unwrap();
        let doc: Vec<u16> = (0..1024).map(|i| i % 64).collect();
        writer.push(&doc, doc.len() * 4).unwrap();
        writer.finish().unwrap();
    }

    fn tiny_run() -> Run {
        Run::new()
            .with_steps(2)
            .with_micro_batch(2)
            .with_accum(2)
            .with_warmup(1)
            .with_decay(1)
            // Exercise the device-side loss aggregation and scalar read.
            .with_log_every(1)
            .with_eval_every(0)
            .with_save_every(0)
            .with_eval_batches(1)
    }

    #[test]
    fn default_tiny_recipe_uses_a_compute_efficient_token_budget() {
        let run = Run::new();
        let tokens = run.steps as u64
            * run.micro_batch as u64
            * run.accum as u64
            * config::Model::tiny().seq_len as u64;

        // A 162.5M-parameter model needs roughly 20 tokens per parameter, not
        // the 96.8 tokens per parameter scheduled by the old 60k-step recipe.
        assert!((3_000_000_000..=3_500_000_000).contains(&tokens), "{tokens} tokens");
    }

    #[test]
    fn old_run_files_without_an_ssd_mode_still_load() {
        let run = Run::new();
        let mut json = serde_json::to_value(&run).unwrap();
        json.as_object_mut().unwrap().remove("ssd_mode");
        json.as_object_mut().unwrap().remove("head_dtype");
        json.as_object_mut().unwrap().remove("ffn_dtype");

        let loaded: Run = serde_json::from_value(json).unwrap();

        assert_eq!(loaded.ssd_mode, None);
        assert_eq!(loaded.head_dtype, None);
        assert_eq!(loaded.ffn_dtype, None);
    }

    #[test]
    fn reported_rate_explains_slow_optimizer_step_progress() {
        let tokens_per_step = 96 * 2_048;
        let throughput = 1_700.0;
        let steps_per_hour = throughput * 3_600.0 / tokens_per_step as f64;
        let (eta_hours, tflops) =
            speed(40, 60_000, tokens_per_step, throughput, config::Model::tiny().flops_per_token());

        assert!((steps_per_hour - 31.1).abs() < 0.1);
        assert!((eta_hours / 24.0 - 80.3).abs() < 0.1);
        assert!((tflops - 1.84).abs() < 0.01);
    }

    #[test]
    fn a_short_run_leaves_a_resumable_checkpoint() {
        let data = tempfile::tempdir().unwrap();
        shards(&data.path().join("train"));
        shards(&data.path().join("valid"));
        let out = tempfile::tempdir().unwrap();

        run(&config::Model::toy(), &tiny_run(), data.path(), out.path(), &Device::default())
            .unwrap();

        assert_eq!(checkpoint::latest(out.path()).unwrap(), checkpoint::dir(out.path(), 2));
    }

    #[test]
    fn a_short_fp16_projection_run_leaves_a_finite_resumable_checkpoint() {
        let data = tempfile::tempdir().unwrap();
        shards(&data.path().join("train"));
        shards(&data.path().join("valid"));
        let out = tempfile::tempdir().unwrap();
        let run = tiny_run()
            .with_head_dtype(Some(config::HeadDtype::F16))
            .with_ffn_dtype(Some(config::FfnDtype::F16));

        super::run(&config::Model::toy(), &run, data.path(), out.path(), &Device::default())
            .unwrap();

        let dir = checkpoint::latest(out.path()).unwrap();
        let state: State =
            serde_json::from_reader(std::fs::File::open(dir.join("state.json")).unwrap()).unwrap();
        assert_eq!(state.step, 2);
        assert_eq!(state.loss_scaler.unwrap().scale(), DynamicLossScaler::INITIAL);
    }

    #[test]
    fn a_run_that_comes_apart_stops_instead_of_finishing_green() {
        // What a diverged run used to do: 4299 steps of NaN, a checkpoint, a
        // grade sheet of flat zeroes, and a green CI tick over the top of it.
        let data = tempfile::tempdir().unwrap();
        shards(&data.path().join("train"));
        shards(&data.path().join("valid"));
        let out = tempfile::tempdir().unwrap();
        let blown = tiny_run().with_lr(1e30);

        let error = run(&config::Model::toy(), &blown, data.path(), out.path(), &Device::default())
            .unwrap_err();

        let Error::Diverged { step, params } = error else {
            panic!("{error}");
        };
        assert_eq!(step, 2);
        // And it says which tensors went, rather than only that something did.
        assert!(!params.is_empty(), "nothing was reported as non-finite");
    }

    #[test]
    fn a_disabled_period_never_comes_due() {
        assert!(!due(41, 0));
    }
}
