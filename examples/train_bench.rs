//! Short, data-independent GPU benchmark for the real tiny-turbo training step.
//!
//! It deliberately uses the preset's own vocabulary and the production
//! optimizer; only corpus I/O and checkpoint-file writes are omitted from the
//! timed region. That vocabulary is 8192 since issue #23; `legacy-turbo` keeps
//! the 32768 of the measurements in `docs/TRAINING_SPEED.md`, so the two are
//! not a paired comparison of shape alone. The first complete optimizer step is warm-up so CubeCL fusion and
//! autotuning do not contaminate the measurement.

mod bench_support;

use std::time::Instant;

use anyhow::Result;
use bench_support::MeasurementSummary;
use burn::prelude::*;
use burn::tensor::{DeviceConfig, FloatDType};
use clap::{Parser, ValueEnum};
use quasar::config::{Model, SsdMode};
use quasar::model::Quasar;
use quasar::train::{DynamicLossScaler, Optim, Run};

#[derive(Parser)]
#[command(about = "Measure a few synchronized tiny-turbo training steps")]
struct Args {
    #[arg(long, value_enum, default_value_t = BenchModel::TinyTurbo)]
    model: BenchModel,
    #[arg(long, default_value_t = 4)]
    micro_batch: usize,
    #[arg(long, default_value_t = 32)]
    accum: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    /// Extend an unstable initial window to this many measured steps.
    #[arg(long, default_value_t = 9)]
    max_steps: usize,
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,
    /// Explicit compute dtype for the output-head GEMM; everything else stays
    /// in the device dtype (fp32 in the paired precision experiment).
    #[arg(long, value_enum)]
    head_dtype: Option<ReducedDtype>,
    /// Explicit compute dtype for the three GEMMs in every FFN. Norms,
    /// SwiGLU elementwise math and the residual stream remain fp32.
    #[arg(long, value_enum)]
    ffn_dtype: Option<ReducedDtype>,
    /// Explicit compute dtype for Mamba input/output projection GEMMs. SSD
    /// coefficients, recurrent state, discretization and norms remain fp32.
    #[arg(long, value_enum)]
    mamba_dtype: Option<ReducedDtype>,
    #[arg(long, value_enum, default_value_t = Ssd::Serial)]
    ssd: Ssd,
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    checkpointing: bool,
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    muon: bool,
    /// Run one synchronized precision diagnostic before warm-up. This is never
    /// included in a measured step.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    precision_diagnostics: bool,
    /// Generate a different deterministic batch for every accumulation pass.
    /// This is for loss-trajectory gates; host-to-device copies make it
    /// intentionally unsuitable for a throughput claim.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    vary_tokens: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Ssd {
    Minimal,
    Serial,
    Recalculated,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReducedDtype {
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BenchModel {
    TinyTurbo,
    LegacyTurbo,
}

impl BenchModel {
    fn config(self) -> Model {
        let mut cfg = Model::tiny_turbo();
        if matches!(self, Self::LegacyTurbo) {
            // The pre-optimization preset remains available so the paired A/B
            // can be reproduced after the wide candidate became the preset. Its
            // vocabulary is part of it: `docs/TRAINING_SPEED.md` measured that
            // A/B at 32768, and issue #23 narrowed only the current preset.
            cfg.d_model = 512;
            cfg.n_layers = 20;
            cfg.attn_heads = 8;
            cfg.vocab_size = 32_768;
        }
        cfg.validate().expect("benchmark model must be valid");
        cfg
    }
}

impl From<Dtype> for FloatDType {
    fn from(value: Dtype) -> Self {
        match value {
            Dtype::F32 => Self::F32,
            Dtype::F16 => Self::F16,
            Dtype::Bf16 => Self::BF16,
        }
    }
}

impl From<ReducedDtype> for FloatDType {
    fn from(value: ReducedDtype) -> Self {
        match value {
            ReducedDtype::F16 => Self::F16,
            ReducedDtype::Bf16 => Self::BF16,
        }
    }
}

impl From<Ssd> for SsdMode {
    fn from(value: Ssd) -> Self {
        match value {
            Ssd::Minimal => Self::Minimal,
            Ssd::Serial => Self::Serial,
            Ssd::Recalculated => Self::Recalculated,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.micro_batch > 0, "micro-batch must be positive");
    assert!(args.accum > 0, "accum must be positive");
    assert!(args.steps > 0, "at least one measured step is required");
    assert!(args.max_steps >= args.steps, "max-steps must be at least steps");

    let cfg = args.model.config();
    let mut base_device = Device::default();
    base_device.configure(DeviceConfig::default().float_dtype(FloatDType::from(args.dtype)))?;
    let device = base_device.clone().autodiff();
    let device = if args.checkpointing { device.gradient_checkpointing() } else { device };
    device.seed(1337);

    let ssd_mode = SsdMode::from(args.ssd);
    let mut model = Quasar::new_with_ssd_and_projection_dtypes(
        &cfg,
        ssd_mode.clone(),
        args.head_dtype.map(FloatDType::from),
        args.ffn_dtype.map(FloatDType::from),
        args.mamba_dtype.map(FloatDType::from),
        &device,
    );
    let run = Run::new()
        .with_micro_batch(args.micro_batch)
        .with_accum(args.accum)
        .with_muon(args.muon)
        .with_checkpointing(args.checkpointing)
        .with_ssd_mode(Some(ssd_mode));
    let mut optim = Optim::new(&run, &model);
    let (input, target) = tokens(&cfg, args.micro_batch, &device);
    let tokens_per_step = args.micro_batch * args.accum * cfg.seq_len;
    let mut loss_scaler = (matches!(args.head_dtype, Some(ReducedDtype::F16))
        || matches!(args.ffn_dtype, Some(ReducedDtype::F16))
        || matches!(args.mamba_dtype, Some(ReducedDtype::F16)))
    .then(|| DynamicLossScaler::new(200));

    println!(
        "bench device={base_device:?} model={:?} dtype={:?} head_dtype={:?} ffn_dtype={:?} mamba_dtype={:?} micro_batch={} accum={} ssd={:?} checkpointing={} muon={} vary_tokens={} tokens/step={tokens_per_step}",
        args.model,
        args.dtype,
        args.head_dtype,
        args.ffn_dtype,
        args.mamba_dtype,
        args.micro_batch,
        args.accum,
        args.ssd,
        args.checkpointing,
        args.muon,
        args.vary_tokens
    );

    if args.precision_diagnostics {
        let loss_scale = loss_scaler.as_ref().map_or(1.0, DynamicLossScaler::scale);
        let diagnostics = model.precision_diagnostics(input.clone(), target.clone(), loss_scale);
        println!(
            "precision activation_min={:.6e} activation_max={:.6e} activation_mean={:.6e} \
             grad_norm={:.6e} loss_scale={:.1} nonfinite_count={} loss={:.6}",
            diagnostics.activation_min,
            diagnostics.activation_max,
            diagnostics.activation_mean,
            diagnostics.grad_norm,
            diagnostics.loss_scale,
            diagnostics.nonfinite_count,
            diagnostics.loss
        );
        anyhow::ensure!(
            diagnostics.nonfinite_count == 0 && diagnostics.loss.is_finite(),
            "precision diagnostic found non-finite values"
        );
    }

    for step in 0..args.warmup {
        let (next, loss, loss_scale) = optimizer_step(
            model,
            &mut optim,
            &input,
            &target,
            &cfg,
            args.accum,
            step,
            args.vary_tokens,
            loss_scaler.as_mut(),
            &device,
        )?;
        model = next;
        println!("warmup {}/{} loss={loss:.4} loss_scale={loss_scale:.0}", step + 1, args.warmup);
    }

    let mut seconds = Vec::with_capacity(args.max_steps);
    for step in 0..args.max_steps {
        let started = Instant::now();
        let (next, loss, loss_scale) = optimizer_step(
            model,
            &mut optim,
            &input,
            &target,
            &cfg,
            args.accum,
            args.warmup + step,
            args.vary_tokens,
            loss_scaler.as_mut(),
            &device,
        )?;
        model = next;
        let elapsed = started.elapsed().as_secs_f64();
        let throughput = tokens_per_step as f64 / elapsed;
        let planned_steps = if step < args.steps { args.steps } else { args.max_steps };
        println!(
            "measured {}/{} loss={loss:.4} loss_scale={loss_scale:.0} seconds={elapsed:.3} throughput={throughput:.0} tok/s",
            step + 1,
            planned_steps
        );
        seconds.push(elapsed);

        if seconds.len() == args.steps {
            let initial = MeasurementSummary::new(&seconds);
            if initial.needs_extended_window() && args.max_steps > args.steps {
                println!(
                    "measurement window extended from {} to {} steps: initial min/max spread {:.2}% exceeds 3%",
                    args.steps, args.max_steps, initial.spread_percent
                );
            } else {
                break;
            }
        }
    }

    let summary = MeasurementSummary::new(&seconds);
    let throughput = tokens_per_step as f64 / summary.median;
    let min_throughput = tokens_per_step as f64 / summary.max;
    let max_throughput = tokens_per_step as f64 / summary.min;
    let tflops = throughput * 3.0 * cfg.flops_per_token() / 1e12;
    println!(
        "result samples={} median_seconds={:.3} min_seconds={:.3} max_seconds={:.3} \
         spread={:.2}% throughput={throughput:.0} tok/s min_throughput={min_throughput:.0} \
         max_throughput={max_throughput:.0} effective={tflops:.2} TFLOP/s",
        summary.samples, summary.median, summary.min, summary.max, summary.spread_percent
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_shape_trades_depth_for_width_without_changing_work() {
        let narrow = BenchModel::LegacyTurbo.config();
        // At the vocabulary the A/B was run on. The claim is about depth
        // against width, and comparing 512 × 20 at 32768 with 640 × 12 at 8192
        // would answer a different question with the same two numbers.
        let wide = Model { vocab_size: narrow.vocab_size, ..BenchModel::TinyTurbo.config() };

        let param_ratio = wide.budget().total as f64 / narrow.budget().total as f64;
        let flop_ratio = wide.flops_per_token() / narrow.flops_per_token();
        let activation_ratio = wide.activations(1).total / narrow.activations(1).total;

        assert!((param_ratio - 1.0).abs() < 0.01, "parameter ratio {param_ratio}");
        assert!((flop_ratio - 1.0).abs() < 0.01, "FLOP ratio {flop_ratio}");
        assert!(activation_ratio < 0.81, "activation ratio {activation_ratio}");
        assert_eq!((wide.d_model, wide.n_layers, wide.attn_heads), (640, 12, 10));
    }

    /// The preset the benchmark actually times is narrower than either arm of
    /// that A/B, and issue #23 is the reason: the vocabulary is the one cut
    /// that takes parameters, FLOPs and activations at once.
    #[test]
    fn the_benchmarked_preset_carries_the_narrowed_vocabulary() {
        let turbo = BenchModel::TinyTurbo.config();
        let legacy = BenchModel::LegacyTurbo.config();

        assert_eq!(turbo.vocab_size, quasar::config::SMALL_VOCAB);
        assert_eq!(legacy.vocab_size, 32_768);
        assert!(turbo.budget().total < legacy.budget().total);
        assert!(turbo.flops_per_token() < legacy.flops_per_token());
        assert!(turbo.activations(1).total < legacy.activations(1).total);
    }

    #[test]
    fn measurement_summary_exposes_unstable_three_step_windows() {
        let summary = MeasurementSummary::new(&[1.00, 1.02, 1.04]);

        assert_eq!(summary.samples, 3);
        assert_eq!(summary.median, 1.02);
        assert_eq!(summary.min, 1.00);
        assert_eq!(summary.max, 1.04);
        assert!(summary.spread_percent > 3.0);
        assert!(summary.needs_extended_window());
    }

    #[test]
    fn measurement_summary_accepts_a_stable_three_step_window() {
        let summary = MeasurementSummary::new(&[1.00, 1.01, 1.02]);

        assert!(summary.spread_percent < 3.0);
        assert!(!summary.needs_extended_window());
    }

    #[test]
    fn trajectory_tokens_are_reproducible_but_change_between_micro_batches() {
        let cfg = BenchModel::TinyTurbo.config();
        let first = token_values_for_stream(&cfg, 32, 7);
        let again = token_values_for_stream(&cfg, 32, 7);
        let next = token_values_for_stream(&cfg, 32, 8);

        assert_eq!(first, again);
        assert_ne!(first, next);
        assert!(first.iter().all(|token| *token >= 0 && *token < cfg.vocab_size as i32));
        assert!(
            first.windows(2).all(|pair| pair[1] == (pair[0] + 1) % cfg.vocab_size as i32),
            "the trajectory must retain one learnable next-token relation"
        );
    }
}

fn tokens(cfg: &Model, batch: usize, device: &Device) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let len = batch * cfg.seq_len;
    let data: Vec<i32> = (0..=len).map(|i| (i % cfg.vocab_size) as i32).collect();
    let input = TensorData::new(data[..len].to_vec(), [batch, cfg.seq_len]);
    let target = TensorData::new(data[1..].to_vec(), [batch, cfg.seq_len]);
    (Tensor::from_data(input, device), Tensor::from_data(target, device))
}

fn tokens_for_stream(
    cfg: &Model,
    batch: usize,
    stream: usize,
    device: &Device,
) -> (Tensor<2, Int>, Tensor<2, Int>) {
    let len = batch * cfg.seq_len;
    let data = token_values_for_stream(cfg, len, stream);
    let input = TensorData::new(data[..len].to_vec(), [batch, cfg.seq_len]);
    let target = TensorData::new(data[1..].to_vec(), [batch, cfg.seq_len]);
    (Tensor::from_data(input, device), Tensor::from_data(target, device))
}

fn token_values_for_stream(cfg: &Model, len: usize, stream: usize) -> Vec<i32> {
    // Every sample teaches the same learnable next-token relation while the
    // coprime stream offset prevents optimizer steps from replaying one batch.
    let vocab = cfg.vocab_size as u64;
    let start = (stream as u64).wrapping_mul(7_919).wrapping_add(1_337) % vocab;
    (0..=len).map(|position| (start + position as u64) % vocab).map(|token| token as i32).collect()
}

#[allow(clippy::too_many_arguments)]
fn optimizer_step(
    mut model: Quasar,
    optim: &mut Optim,
    input: &Tensor<2, Int>,
    target: &Tensor<2, Int>,
    cfg: &Model,
    accum: usize,
    optimizer_step: usize,
    vary_tokens: bool,
    loss_scaler: Option<&mut DynamicLossScaler>,
    device: &Device,
) -> Result<(Quasar, f32, f64)> {
    let mut logged_loss = None;
    let loss_scale = loss_scaler.as_ref().map_or(1.0, |scaler| scaler.scale());
    for micro_step in 0..accum {
        let (input, target) = if vary_tokens {
            tokens_for_stream(cfg, input.dims()[0], optimizer_step * accum + micro_step, device)
        } else {
            (input.clone(), target.clone())
        };
        let loss = model.loss(input, target);
        let nll = loss.nll.clone().detach();
        logged_loss = Some(match logged_loss.take() {
            Some(total) => total + nll,
            None => nll,
        });
        let grads = loss.total.div_scalar(accum as f64).mul_scalar(loss_scale).backward();
        optim.accumulate(&model, grads);
    }
    model = match loss_scaler {
        Some(scaler) => {
            let (model, finite) = optim.step_scaled(3e-3, model, loss_scale);
            scaler.update(finite);
            anyhow::ensure!(finite, "dynamic loss scaler found non-finite gradients");
            model
        }
        None => optim.step(3e-3, model),
    };
    device.sync()?;
    let loss = logged_loss.unwrap().div_scalar(accum as f64).into_scalar::<f32>();
    anyhow::ensure!(loss.is_finite(), "training produced a non-finite loss");
    Ok((model, loss, loss_scale))
}
