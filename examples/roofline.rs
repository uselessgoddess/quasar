//! Dense-matmul roofline probe for the exact Burn/CubeCL backend used to train.
//!
//! Inputs are created once, outside the timed region. Every sample dispatches
//! one GEMM and synchronizes the device so the reported duration is GPU work
//! plus the framework's launch cost, not merely command submission.

mod bench_support;

use std::time::Instant;

use anyhow::Result;
use bench_support::MeasurementSummary;
use burn::prelude::*;
use burn::tensor::{DeviceConfig, FloatDType};
use clap::{Parser, ValueEnum};

#[derive(Parser)]
#[command(about = "Measure square dense-matmul throughput through Burn/CubeCL")]
struct Args {
    #[arg(long, default_value_t = 4096)]
    size: usize,
    #[arg(long, value_enum, default_value_t = Dtype::F32)]
    dtype: Dtype,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    steps: usize,
    /// Extend an unstable initial window to this many measured samples.
    #[arg(long, default_value_t = 9)]
    max_steps: usize,
    /// Hardware peak used only for the reported utilization percentage.
    #[arg(long)]
    peak_tflops: Option<f64>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn float(self) -> FloatDType {
        match self {
            Self::F32 => FloatDType::F32,
            Self::F16 => FloatDType::F16,
            Self::Bf16 => FloatDType::BF16,
        }
    }

    fn rdna4_peak_tflops(self) -> f64 {
        match self {
            Self::F32 => 48.0,
            Self::F16 | Self::Bf16 => 97.0,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    assert!(args.size > 0, "size must be positive");
    assert!(args.warmup > 0, "at least one warm-up is required");
    assert!(args.steps >= 3, "at least three measured samples are required");
    assert!(args.max_steps >= args.steps, "max-steps must be at least steps");

    // Keep the device default at fp32. Reduced-precision probes use explicit
    // per-tensor casts, which is the same mechanism needed for mixed precision
    // around selected GEMMs without moving norms, SSD, or optimizer state.
    let mut device = Device::default();
    device.configure(DeviceConfig::default().float_dtype(FloatDType::F32))?;
    device.seed(1337);

    let dtype = args.dtype.float();
    let lhs: Tensor<2> = Tensor::ones([args.size, args.size], &device);
    let rhs: Tensor<2> = Tensor::ones([args.size, args.size], &device);
    let lhs = lhs.cast(dtype);
    let rhs = rhs.cast(dtype);
    device.sync()?;

    println!(
        "roofline device={device:?} dtype={:?} m={} n={} k={} warmup={} initial_samples={} max_samples={}",
        args.dtype, args.size, args.size, args.size, args.warmup, args.steps, args.max_steps
    );

    for step in 0..args.warmup {
        let output = lhs.clone().matmul(rhs.clone());
        device.sync()?;
        drop(output);
        println!("warmup {}/{}", step + 1, args.warmup);
    }

    let mut seconds = Vec::with_capacity(args.max_steps);
    for step in 0..args.max_steps {
        let started = Instant::now();
        let output = lhs.clone().matmul(rhs.clone());
        device.sync()?;
        let elapsed = started.elapsed().as_secs_f64();
        drop(output);
        let tflops = gemm_tflops(args.size, elapsed);
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
                    "measurement window extended from {} to {} samples: initial min/max spread {:.2}% exceeds 3%",
                    args.steps, args.max_steps, initial.spread_percent
                );
            } else {
                break;
            }
        }
    }

    let summary = MeasurementSummary::new(&seconds);
    let median_tflops = gemm_tflops(args.size, summary.median);
    let min_tflops = gemm_tflops(args.size, summary.max);
    let max_tflops = gemm_tflops(args.size, summary.min);
    let peak_tflops = args.peak_tflops.unwrap_or_else(|| args.dtype.rdna4_peak_tflops());
    println!(
        "result dtype={:?} shape={}x{}x{} samples={} median_seconds={:.6} \
         min_seconds={:.6} max_seconds={:.6} spread={:.2}% throughput={median_tflops:.2} \
         TFLOP/s min_throughput={min_tflops:.2} max_throughput={max_tflops:.2} \
         peak={peak_tflops:.1} TFLOP/s utilization={:.1}%",
        args.dtype,
        args.size,
        args.size,
        args.size,
        summary.samples,
        summary.median,
        summary.min,
        summary.max,
        summary.spread_percent,
        median_tflops / peak_tflops * 100.0
    );
    Ok(())
}

fn gemm_tflops(size: usize, seconds: f64) -> f64 {
    2.0 * (size as f64).powi(3) / seconds / 1e12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_matmul_counts_multiply_and_add() {
        assert!((gemm_tflops(1_000, 0.002) - 1.0).abs() < f64::EPSILON);
    }
}
