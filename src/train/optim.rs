//! The optimizer, and which half of the model each part of it sees.
//!
//! AdamW keeps two fp32 moments per parameter, so weights, gradients and states
//! come to 16 B/param — 16.65 GiB for `base` before a single activation is
//! allocated, which is why `base` did not fit. Muon keeps one momentum buffer
//! instead, 12 B/param, and orthogonalises the update, which is the better
//! optimizer for a hidden matrix anyway.
//!
//! What Muon cannot do is anything that is not a matrix: its step asserts a 2-D
//! tensor, so norms and biases are out by construction, and the embedding and
//! the head are out by meaning — their rows are tokens, and a step touches only
//! the tokens in the batch, so orthogonalising over the vocabulary would mix
//! rows that were never seen. Those stay on AdamW. Splitting them is what
//! [`GradientsParams::from_params`] is for: it *removes* the parameters it
//! matches, so the `from_module` that follows collects exactly the remainder.

use std::path::Path;

use burn::grad_clipping::GradientClippingConfig;
use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{
    AdamWConfig, AdjustLrFn, GradientsAccumulator, GradientsParams, ModuleOptimizer, MuonConfig,
};
use burn::store::{ModuleSnapshot, RecordError};
use burn::tensor::{Bool, Gradients, Tensor};
use serde::{Deserialize, Serialize};

use crate::model::Quasar;
use crate::train::Run;

/// Dynamic loss scale for the measured fp16 projection paths.
///
/// A non-finite step is discarded before either optimizer sees it. The scale
/// then halves and the caller retries the same deterministic batch. A long
/// clean window grows it again, bounded at both ends.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DynamicLossScaler {
    scale: f64,
    finite_steps: usize,
    growth_interval: usize,
}

impl DynamicLossScaler {
    pub const INITIAL: f64 = 1024.0;
    pub const MIN: f64 = 1.0;
    pub const MAX: f64 = 65536.0;

    pub fn new(growth_interval: usize) -> Self {
        assert!(growth_interval > 0, "growth interval must be positive");
        Self { scale: Self::INITIAL, finite_steps: 0, growth_interval }
    }

    pub fn scale(&self) -> f64 {
        self.scale
    }

    pub fn update(&mut self, finite: bool) {
        if finite {
            self.finite_steps += 1;
            if self.finite_steps == self.growth_interval {
                self.scale = (self.scale * 2.0).min(Self::MAX);
                self.finite_steps = 0;
            }
        } else {
            self.scale = (self.scale / 2.0).max(Self::MIN);
            self.finite_steps = 0;
        }
    }
}

/// Both optimizers and the accumulator each of them feeds from.
pub struct Optim {
    adamw: ModuleOptimizer,
    adamw_grads: GradientsAccumulator<Quasar>,
    muon: ModuleOptimizer,
    muon_grads: GradientsAccumulator<Quasar>,
    /// The parameters Muon steps; empty when it is switched off, which is the
    /// only thing that distinguishes an AdamW-only run.
    hidden: Vec<ParamId>,
}

impl Optim {
    pub fn new(run: &Run, model: &Quasar) -> Self {
        let clip = GradientClippingConfig::Norm(run.clip);
        Self {
            adamw: AdamWConfig::new()
                .with_weight_decay(run.weight_decay)
                .with_grad_clipping(Some(clip.clone()))
                .init(),
            adamw_grads: GradientsAccumulator::new(),
            muon: MuonConfig::new()
                .with_weight_decay(Some(WeightDecayConfig::new(run.weight_decay)))
                // Moonshot's scaling gives an orthogonalised update the same RMS
                // as an AdamW one, so the schedule stays as it was tuned.
                .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
                .init()
                .with_grad_clipping(clip.init()),
            muon_grads: GradientsAccumulator::new(),
            hidden: if run.muon { hidden(model) } else { Vec::new() },
        }
    }

    /// Split one micro-batch's gradients between the two optimizers.
    pub fn accumulate(&mut self, model: &Quasar, mut grads: Gradients) {
        if !self.hidden.is_empty() {
            let matrices = GradientsParams::from_params(&mut grads, model, &self.hidden);
            self.muon_grads.accumulate(model, matrices);
        }
        let rest = GradientsParams::from_module(&mut grads, model);
        self.adamw_grads.accumulate(model, rest);
    }

    /// Apply everything accumulated since the last step.
    pub fn step(&mut self, lr: f64, model: Quasar) -> Quasar {
        let muon_grads = self.muon_grads.grads();
        let adamw_grads = self.adamw_grads.grads();
        self.apply(lr, model, muon_grads, adamw_grads)
    }

    /// Unscale accumulated gradients once per optimizer step, reject
    /// non-finite values, and otherwise apply the same Muon/AdamW split.
    ///
    /// Delaying the unscale until after accumulation avoids adding a kernel
    /// for every parameter to every micro-batch. The single all-finite scalar
    /// read is also outside all forward/backward work.
    pub fn step_scaled(&mut self, lr: f64, model: Quasar, loss_scale: f64) -> (Quasar, bool) {
        assert!(
            loss_scale.is_finite() && loss_scale > 0.0,
            "loss scale must be finite and positive"
        );
        let mut finite = Vec::new();
        let muon_grads = unscale(self.muon_grads.grads(), &model, loss_scale, &mut finite);
        let adamw_grads = unscale(self.adamw_grads.grads(), &model, loss_scale, &mut finite);
        let all_finite = !finite.is_empty() && Tensor::cat(finite, 0).all().into_scalar::<bool>();

        if all_finite {
            (self.apply(lr, model, muon_grads, adamw_grads), true)
        } else {
            (model, false)
        }
    }

    fn apply(
        &mut self,
        lr: f64,
        model: Quasar,
        muon_grads: GradientsParams,
        adamw_grads: GradientsParams,
    ) -> Quasar {
        let model = match self.hidden.is_empty() {
            true => model,
            false => self.muon.step(lr, model, muon_grads),
        };
        self.adamw.step(lr, model, adamw_grads)
    }

    pub fn save(&self, dir: &Path) -> Result<(), RecordError> {
        self.adamw.save(dir.join("optim.bpk"))?;
        self.muon.save(dir.join("muon.bpk"))
    }

    pub fn load(self, dir: &Path) -> Result<Self, RecordError> {
        let Self { adamw, muon, adamw_grads, muon_grads, hidden } = self;
        Ok(Self {
            adamw: adamw.load(dir.join("optim.bpk"))?,
            muon: muon.load(dir.join("muon.bpk"))?,
            adamw_grads,
            muon_grads,
            hidden,
        })
    }
}

fn unscale(
    mut grads: GradientsParams,
    model: &Quasar,
    loss_scale: f64,
    finite: &mut Vec<Tensor<1, Bool>>,
) -> GradientsParams {
    model.visit(&mut GradientUnscaler { grads: &mut grads, inverse: 1.0 / loss_scale, finite });
    grads
}

struct GradientUnscaler<'a> {
    grads: &'a mut GradientsParams,
    inverse: f64,
    finite: &'a mut Vec<Tensor<1, Bool>>,
}

impl ModuleVisitor for GradientUnscaler<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(grad) = self.grads.remove::<D>(param.id) else {
            return;
        };
        let grad = grad.mul_scalar(self.inverse);
        self.finite.push(grad.clone().is_finite().all());
        self.grads.register(param.id, grad);
    }
}

/// The parameters Muon may step: the two-dimensional weights that are not the
/// vocabulary. That is the projections of every mixer and feed-forward, which is
/// the overwhelming majority of the parameters and so of the optimizer state.
fn hidden(model: &Quasar) -> Vec<ParamId> {
    model
        .collect(None, None, false)
        .iter()
        .filter(|tensor| tensor.shape.num_dims() == 2 && !vocab(&tensor.full_path()))
        .filter_map(|tensor| tensor.tensor_id)
        .collect()
}

/// Whether a parameter path names a matrix whose rows are tokens.
fn vocab(path: &str) -> bool {
    path.contains("embed") || path.contains("head")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    use crate::config;

    fn model() -> Quasar {
        Quasar::new(&config::Model::toy(), &Device::default())
    }

    #[test]
    fn muon_takes_the_matrices_and_leaves_the_vocabulary() {
        let model = model();
        let hidden = hidden(&model);

        for tensor in model.collect(None, None, false) {
            let matrix = tensor.shape.num_dims() == 2;
            let path = tensor.full_path();
            let taken = hidden.contains(&tensor.tensor_id.unwrap());
            assert_eq!(taken, matrix && !vocab(&path), "{path} {:?}", tensor.shape);
        }
        assert!(!hidden.is_empty(), "the toy model has projections to orthogonalise");
    }

    /// The split is only correct if it is a partition: a matrix that reaches
    /// neither optimizer silently stops training, and one that reaches Muon
    /// without being a matrix panics inside the Newton-Schulz iteration.
    #[test]
    fn a_step_moves_both_halves_of_the_split() {
        let (device, run) = (Device::default().autodiff(), Run::new());
        let model = Quasar::new(&config::Model::toy(), &device);
        let before = model.clone().collect(None, None, false);
        let mut optim = Optim::new(&run, &model);

        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);
        let grads = model.loss(tokens.clone(), tokens).total.backward();
        optim.accumulate(&model, grads);
        let model = optim.step(1e-2, model);

        let after = model.collect(None, None, false);
        assert_eq!(before.len(), after.len());
        // Every matrix has a gradient after one forward, whichever optimizer
        // owns it — the vocabulary through AdamW, the rest through Muon.
        for (before, after) in before.iter().zip(&after).filter(|(t, _)| t.shape.num_dims() == 2) {
            let was = before.to_data().unwrap().to_vec::<f32>().unwrap();
            let now = after.to_data().unwrap().to_vec::<f32>().unwrap();
            let step = was.iter().zip(&now).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(step > 0.0, "{} never moved", before.full_path());
        }
    }

    #[test]
    fn scaled_step_matches_the_unscaled_optimizer_path() {
        let (device, run) = (Device::default().autodiff(), Run::new());
        let reference = Quasar::new(&config::Model::toy(), &device);
        let scaled = reference.clone();
        let tokens = Tensor::<2, burn::tensor::Int>::zeros([2, 8], &device);

        let mut reference_optim = Optim::new(&run, &reference);
        let reference_grads = reference.loss(tokens.clone(), tokens.clone()).total.backward();
        reference_optim.accumulate(&reference, reference_grads);
        let reference = reference_optim.step(1e-2, reference);

        let mut scaled_optim = Optim::new(&run, &scaled);
        let scale = 1024.0;
        let scaled_grads = scaled.loss(tokens.clone(), tokens).total.mul_scalar(scale).backward();
        scaled_optim.accumulate(&scaled, scaled_grads);
        let (scaled, finite) = scaled_optim.step_scaled(1e-2, scaled, scale);
        assert!(finite);

        for (reference, scaled) in
            reference.collect(None, None, false).iter().zip(scaled.collect(None, None, false))
        {
            let reference_values = reference.to_data().unwrap().to_vec::<f32>().unwrap();
            let scaled_values = scaled.to_data().unwrap().to_vec::<f32>().unwrap();
            let delta = reference_values
                .iter()
                .zip(scaled_values)
                .map(|(reference, scaled)| (reference - scaled).abs())
                .fold(0.0, f32::max);
            assert!(delta < 1e-6, "{} delta {delta}", reference.full_path());
        }
    }

    #[test]
    fn scaled_step_rejects_nonfinite_gradients_without_moving_the_model() {
        let (device, run) = (Device::default().autodiff(), Run::new());
        let model = Quasar::new(&config::Model::toy(), &device);
        let before = model.clone().collect(None, None, false);
        let tokens = Tensor::<2, burn::tensor::Int>::zeros([2, 8], &device);
        let mut optim = Optim::new(&run, &model);

        let grads = model.loss(tokens.clone(), tokens).total.mul_scalar(f64::NAN).backward();
        optim.accumulate(&model, grads);
        let (model, finite) = optim.step_scaled(1e-2, model, 1024.0);

        assert!(!finite);
        for (before, after) in before.iter().zip(model.collect(None, None, false)) {
            assert_eq!(
                before.to_data().unwrap().to_vec::<f32>().unwrap(),
                after.to_data().unwrap().to_vec::<f32>().unwrap(),
                "{} moved after a rejected step",
                before.full_path()
            );
        }
    }

    #[test]
    fn loss_scaler_backs_off_and_grows_after_a_clean_window() {
        let mut scaler = DynamicLossScaler::new(2);
        assert_eq!(scaler.scale(), 1024.0);

        scaler.update(false);
        assert_eq!(scaler.scale(), 512.0);
        scaler.update(true);
        assert_eq!(scaler.scale(), 512.0);
        scaler.update(true);
        assert_eq!(scaler.scale(), 1024.0);
    }
}
