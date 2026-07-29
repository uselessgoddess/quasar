//! Fused recurrent single-SSD scan for the production MIMO-rank-one case.
//!
//! The forward kernel replaces the five materialized serial SSD stages with the
//! exact token recurrence
//!
//! `pre = exp(da) * state; y = C * (pre + gamma * B * v);`
//! `state = pre + scale * B * v`.
//!
//! Its custom backward needs the state the forward pass visited before each
//! token, and gets it by replaying the recurrence forward from a checkpoint
//! taken every eight tokens, so an `O(tokens * state_rank)` reduction buffer
//! and one-eighth of a state history replace the full one. CubeCL backends use
//! the operation by default; set `BURN_MAMBA_FUSED_SINGLE_SCAN=0` to select the
//! five-stage tensor reference.
//!
//! It must not instead divide the state back out by `exp(-da)`, which is the
//! cheaper reconstruction and the one this scan shipped with. `da = Δ·A` is
//! learned and negative, so `exp(-da)` is a multiplier above one applied once
//! per token: it magnifies the fp32 rounding error of the checkpoint by
//! `exp(8|da|)` over a block, on top of the cancellation in the subtraction
//! that precedes it. That is 1e-7 at the `|da| ≈ 0.5` of an initialised model
//! and unbounded by the time one has trained -- see
//! `experiments/inverse_decay.rs` and issue #23, where it stopped `tiny-turbo`
//! around step 70 with non-finite gradients from a finite loss.

/// Tokens between retained states. Checkpoint `c` is the state the block of
/// tokens `[c · INTERVAL, (c+1) · INTERVAL)` opens with, so checkpoint zero is
/// the initial state and the last one -- index `tokens.div_ceil(INTERVAL)` --
/// is the final state the operation returns.
pub(super) const RECONSTRUCTION_INTERVAL: usize = 8;

mod single_ssd_scan;

#[cfg(feature = "autodiff")]
mod backward;

#[cfg(feature = "cubecl")]
mod cube;

#[cfg(feature = "fusion")]
mod fusion;

pub use single_ssd_scan::{Mamba3SingleSsdScanBackendExt, single_ssd_scan};

#[cfg(all(test, feature = "_dev-test"))]
pub(crate) use single_ssd_scan::single_ssd_scan_reference;
