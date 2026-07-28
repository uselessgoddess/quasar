//! Backend go/no-go for the model's largest projection shape.
//!
//! The timed unit is one bias-free 640×32768 linear layer, including its
//! forward pass and both matrix-multiply gradients.  A small untimed fp32/fp16
//! comparison exercises the same output width before any performance sample.

mod bench_support;

use std::time::Instant;

use anyhow::Result;
use bench_support::MeasurementSummary;
use burn::prelude::*;
use burn::tensor::{DeviceConfig, FloatDType};
use clap::Parser;

const LOSS_SCALE: f64 = 1024.0;

#[derive(Parser)]
#[command(about = "Measure a 640x32768 linear forward/backward backend spike")]
struct Args {
    /// Number of flattened tokens presented to the projection.
    #[arg(long, default_value_t = 4096)]
    rows: usize,
    #[arg(long, default_value_t = 640)]
    input_features: usize,
    #[arg(long, default_value_t = 32768)]
    output_features: usize,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    /// Extend an unstable initial window to this many measured samples.
    #[arg(long, default_value_t = 9)]
    max_steps: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.rows > 0, "rows must be positive");
    assert!(args.input_features > 0, "input-features must be positive");
    assert!(args.output_features > 0, "output-features must be positive");
    assert!(args.warmup > 0, "at least one warm-up is required");
    assert!(args.steps >= 3, "at least three measured samples are required");
    assert!(args.max_steps >= args.steps, "max-steps must be at least steps");

    let mut base_device = Device::default();
    base_device.configure(DeviceConfig::default().float_dtype(FloatDType::F32))?;
    let device = base_device.clone().autodiff();
    device.seed(1337);

    correctness_gate(args.input_features, args.output_features, &device)?;

    let input: Tensor<2> =
        Tensor::<2>::ones([args.rows, args.input_features], &device).require_grad();
    let weight: Tensor<2> =
        Tensor::<2>::ones([args.input_features, args.output_features], &device).require_grad();
    device.sync()?;

    println!(
        "linear-backend device={base_device:?} dtype=F16 rows={} shape={}x{} \
         warmup={} initial_samples={} max_samples={} loss_scale={LOSS_SCALE:.0}",
        args.rows,
        args.input_features,
        args.output_features,
        args.warmup,
        args.steps,
        args.max_steps
    );

    for step in 0..args.warmup {
        forward_backward(input.clone(), weight.clone(), &device)?;
        println!("warmup {}/{}", step + 1, args.warmup);
    }

    let mut seconds = Vec::with_capacity(args.max_steps);
    for step in 0..args.max_steps {
        let started = Instant::now();
        forward_backward(input.clone(), weight.clone(), &device)?;
        let elapsed = started.elapsed().as_secs_f64();
        let tflops =
            linear_training_tflops(args.rows, args.input_features, args.output_features, elapsed);
        let planned_steps = if step < args.steps { args.steps } else { args.max_steps };
        println!(
            "measured {}/{} seconds={elapsed:.6} throughput={tflops:.2} TFLOP/s",
            step + 1,
            planned_steps
        );
        seconds.push(elapsed);

        if seconds.len() == args.steps {
            let initial = MeasurementSummary::new(&seconds);
            if initial.needs_extended_window() && args.max_steps > args.steps {
                println!(
                    "measurement window extended from {} to {} samples: initial min/max spread \
                     {:.2}% exceeds 3%",
                    args.steps, args.max_steps, initial.spread_percent
                );
            } else {
                break;
            }
        }
    }

    let summary = MeasurementSummary::new(&seconds);
    let median_tflops = linear_training_tflops(
        args.rows,
        args.input_features,
        args.output_features,
        summary.median,
    );
    let min_tflops =
        linear_training_tflops(args.rows, args.input_features, args.output_features, summary.max);
    let max_tflops =
        linear_training_tflops(args.rows, args.input_features, args.output_features, summary.min);
    println!(
        "result dtype=F16 rows={} shape={}x{} samples={} median_seconds={:.6} \
         min_seconds={:.6} max_seconds={:.6} spread={:.2}% throughput={median_tflops:.2} \
         TFLOP/s min_throughput={min_tflops:.2} max_throughput={max_tflops:.2}",
        args.rows,
        args.input_features,
        args.output_features,
        summary.samples,
        summary.median,
        summary.min,
        summary.max,
        summary.spread_percent
    );
    Ok(())
}

fn correctness_gate(input_features: usize, output_features: usize, device: &Device) -> Result<()> {
    // Thirty-two rows keep the 1024x-scaled fp16 input gradient below 65,504
    // while still exercising the full output width.  Non-uniform data catches
    // layout/transposition mistakes that an all-one matrix would hide.
    let correctness_rows = 32;
    let input_data = TensorData::new(
        patterned_values(correctness_rows * input_features, 31, 15.0, 64.0),
        [correctness_rows, input_features],
    );
    let weight_data = TensorData::new(
        patterned_values(input_features * output_features, 29, 14.0, 128.0),
        [input_features, output_features],
    );
    let reference_input = Tensor::<2>::from_data(input_data.clone(), device).require_grad();
    let reference_weight = Tensor::<2>::from_data(weight_data.clone(), device).require_grad();
    let mixed_input_master = Tensor::<2>::from_data(input_data, device).require_grad();
    let mixed_weight_master = Tensor::<2>::from_data(weight_data, device).require_grad();
    let mixed_input = mixed_input_master.clone().cast(FloatDType::F16);
    let mixed_weight = mixed_weight_master.clone().cast(FloatDType::F16);

    let reference_output = reference_input.clone().matmul(reference_weight.clone());
    let mixed_output = mixed_input.clone().matmul(mixed_weight.clone()).cast(FloatDType::F32);
    let output_error =
        (reference_output.clone() - mixed_output.clone()).abs().max().into_scalar::<f32>();

    let reference_grads = reference_output.powi_scalar(2).mean().backward();
    let mixed_grads = mixed_output.clone().powi_scalar(2).mean().mul_scalar(LOSS_SCALE).backward();
    let reference_input_grad =
        reference_input.grad(&reference_grads).expect("reference input gradient");
    let mixed_input_grad = mixed_input_master
        .grad(&mixed_grads)
        .expect("mixed input gradient")
        .div_scalar(LOSS_SCALE)
        .cast(FloatDType::F32);
    let reference_weight_grad =
        reference_weight.grad(&reference_grads).expect("reference weight gradient");
    let mixed_weight_grad = mixed_weight_master
        .grad(&mixed_grads)
        .expect("mixed weight gradient")
        .div_scalar(LOSS_SCALE)
        .cast(FloatDType::F32);
    let input_grad_error =
        (reference_input_grad - mixed_input_grad.clone()).abs().max().into_scalar::<f32>();
    let weight_grad_error =
        (reference_weight_grad - mixed_weight_grad.clone()).abs().max().into_scalar::<f32>();

    let activation_min = mixed_output.clone().min().into_scalar::<f32>();
    let activation_max = mixed_output.clone().max().into_scalar::<f32>();
    let activation_mean = mixed_output.clone().mean().into_scalar::<f32>();
    let grad_norm = (mixed_input_grad.clone() * mixed_input_grad).sum().sqrt().into_scalar::<f32>();
    let nonfinite_count = nonfinite_count(mixed_output)
        + nonfinite_count(mixed_weight_grad)
        + usize::from(!output_error.is_finite())
        + usize::from(!input_grad_error.is_finite())
        + usize::from(!weight_grad_error.is_finite());

    println!(
        "precision activation_min={activation_min:.6e} activation_max={activation_max:.6e} \
         activation_mean={activation_mean:.6e} grad_norm={grad_norm:.6e} \
         loss_scale={LOSS_SCALE:.1} nonfinite_count={nonfinite_count} \
         output_max_error={output_error:.6e} input_grad_max_error={input_grad_error:.6e} \
         weight_grad_max_error={weight_grad_error:.6e}"
    );
    anyhow::ensure!(nonfinite_count == 0, "linear spike produced non-finite values");
    anyhow::ensure!(output_error <= 1e-2, "maximum output error {output_error}");
    anyhow::ensure!(input_grad_error <= 1e-2, "maximum input-gradient error {input_grad_error}");
    anyhow::ensure!(weight_grad_error <= 1e-2, "maximum weight-gradient error {weight_grad_error}");
    Ok(())
}

fn patterned_values(length: usize, period: usize, center: f32, scale: f32) -> Vec<f32> {
    (0..length).map(|i| ((i % period) as f32 - center) / scale).collect()
}

fn forward_backward(input: Tensor<2>, weight: Tensor<2>, device: &Device) -> Result<()> {
    let output = input
        .clone()
        .cast(FloatDType::F16)
        .matmul(weight.clone().cast(FloatDType::F16))
        .cast(FloatDType::F32);
    let grads = output.powi_scalar(2).mean().mul_scalar(LOSS_SCALE).backward();
    let input_grad = input.grad(&grads).expect("input gradient");
    let weight_grad = weight.grad(&grads).expect("weight gradient");
    device.sync()?;
    drop(input_grad);
    drop(weight_grad);
    Ok(())
}

fn nonfinite_count<const D: usize>(tensor: Tensor<D>) -> usize {
    tensor.is_finite().bool_not().int().sum().into_scalar::<i64>() as usize
}

fn linear_training_tflops(rows: usize, input: usize, output: usize, seconds: f64) -> f64 {
    // Forward, input-gradient and weight-gradient GEMMs, each with one multiply
    // and one add per inner-dimension element.
    6.0 * rows as f64 * input as f64 * output as f64 / seconds / 1e12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_training_counts_three_multiply_add_gemms() {
        assert!((linear_training_tflops(1_000, 1_000, 1_000, 0.006) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn correctness_pattern_is_non_uniform_and_centered() {
        let values = patterned_values(5, 3, 1.0, 2.0);
        assert_eq!(values, vec![-0.5, 0.0, 0.5, -0.5, 0.0]);
    }
}
