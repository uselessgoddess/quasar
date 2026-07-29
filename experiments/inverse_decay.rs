//! How far the reverse-division state reconstruction of the fused rank-one SSD
//! scan drifts from the states the forward pass actually visited.
//!
//! The forward recurrence is `s(t) = exp(da) * s(t-1) + scale * b * v`. The
//! custom backward of `vendor/burn-mamba/src/mamba3/single_ssd_scan` walks it in
//! reverse and recovers `s(t-1)` as `exp(-da) * (s(t) - scale * b * v)` from a
//! checkpoint taken every `RECONSTRUCTION_INTERVAL` tokens. Both steps lose
//! precision: the subtraction cancels when the decay is strong, and the
//! multiplication by `exp(-da)` then magnifies whatever is left.
//!
//! `rustc -O experiments/inverse_decay.rs -o /tmp/inverse_decay && /tmp/inverse_decay`

const INTERVAL: usize = 8;

/// A deterministic value in `[-1, 1]` -- no dependencies, so this file compiles
/// with plain `rustc`.
fn noise(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (*seed >> 8) as f32 / (1u32 << 23) as f32 - 1.0
}

fn main() {
    println!("{:>8} {:>14} {:>14} {:>14}", "|da|", "state", "reverse", "rel err");
    for da_magnitude in [0.5f32, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 90.0] {
        let da = -da_magnitude;
        let mut seed = 23u32;
        // Forward over one reconstruction block, from a checkpoint at t = -1.
        let mut state = 0.7f32;
        let mut injections = Vec::new();
        for _ in 0..INTERVAL {
            let injection = 0.5 * noise(&mut seed);
            injections.push(injection);
            state = da.exp() * state + injection;
        }
        let closing = state;
        // Reverse: what the backward believes the opening state was.
        let mut reverse = closing;
        for injection in injections.iter().rev() {
            reverse = (-da).exp() * (reverse - injection);
        }
        // Forward again from the same checkpoint: what it actually was.
        let opening = 0.7f32;
        let error = ((reverse - opening) / opening).abs();
        println!("{da_magnitude:>8.1} {opening:>14.6} {reverse:>14.6} {error:>14.3e}");
    }
}
