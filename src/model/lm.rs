//! The language model: embedding, a stack of [`Block`]s, and the head.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::FloatDType;
use burn::tensor::activation::log_softmax;
use burn::tensor::module::linear;

use crate::config;
use crate::model::{Block, init};

/// A quasar model, built from [`config::Model`].
#[derive(Module, Debug)]
pub struct Quasar {
    embed: Embedding,
    blocks: Vec<Block>,
    norm: RmsNorm,
    /// `None` when embeddings are tied, in which case the embedding matrix is
    /// transposed into the head at every forward.
    head: Option<Linear>,
    z_loss: f64,
    /// Optional compute dtype for the output projection only. Parameters,
    /// optimizer state, normalized hidden states, logits and loss stay fp32.
    #[module(skip)]
    head_dtype: Option<FloatDType>,
}

impl Quasar {
    pub fn new(cfg: &config::Model, device: &Device) -> Self {
        Self::new_with_ssd(cfg, config::SsdMode::default(), device)
    }

    /// Build the same model with an explicit SSD memory/speed tradeoff.
    ///
    /// `ssd_mode` changes only which mathematically equivalent burn-mamba
    /// backward implementation runs; it does not change parameters or records.
    pub fn new_with_ssd(cfg: &config::Model, ssd_mode: config::SsdMode, device: &Device) -> Self {
        Self::new_with_ssd_and_head_dtype(cfg, ssd_mode, None, device)
    }

    /// Build an experiment that casts only the output-head GEMM.
    pub fn new_with_head_dtype(
        cfg: &config::Model,
        head_dtype: FloatDType,
        device: &Device,
    ) -> Self {
        Self::new_with_ssd_and_head_dtype(cfg, config::SsdMode::default(), Some(head_dtype), device)
    }

    /// Build with independent SSD and output-head execution choices.
    pub fn new_with_ssd_and_head_dtype(
        cfg: &config::Model,
        ssd_mode: config::SsdMode,
        head_dtype: Option<FloatDType>,
        device: &Device,
    ) -> Self {
        cfg.validate().expect("model config is invalid");
        Self {
            embed: EmbeddingConfig::new(cfg.vocab_size, cfg.d_model)
                .with_initializer(init::normal())
                .init(device),
            blocks: (0..cfg.n_layers)
                .map(|i| Block::new(cfg, i, ssd_mode.clone(), device))
                .collect(),
            norm: RmsNormConfig::new(cfg.d_model).init(device),
            head: (!cfg.tied_embeddings).then(|| {
                LinearConfig::new(cfg.d_model, cfg.vocab_size)
                    .with_bias(false)
                    .with_initializer(init::normal())
                    .init(device)
            }),
            z_loss: cfg.z_loss,
            head_dtype,
        }
    }

    /// `[batch, seq] -> [batch, seq, vocab]`.
    pub fn forward(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        self.project(self.hidden(tokens))
    }

    fn hidden(&self, tokens: Tensor<2, Int>) -> Tensor<3> {
        let x = self.blocks.iter().fold(self.embed.forward(tokens), |x, b| b.forward(x));
        self.norm.forward(x)
    }

    fn project(&self, x: Tensor<3>) -> Tensor<3> {
        match self.head_dtype {
            Some(dtype) => match &self.head {
                Some(head) => linear(
                    x.cast(dtype),
                    head.weight.val().cast(dtype),
                    head.bias.as_ref().map(|bias| bias.val().cast(dtype)),
                )
                .cast(FloatDType::F32),
                None => tied_head(x, self.embed.weight.val(), Some(dtype)),
            },
            None => match &self.head {
                Some(head) => head.forward(x),
                None => tied_head(x, self.embed.weight.val(), None),
            },
        }
    }

    /// One untimed forward/backward diagnostic for a precision experiment.
    ///
    /// Reductions intentionally synchronize the device, so callers must keep
    /// this outside any throughput measurement.
    pub fn precision_diagnostics(
        &self,
        tokens: Tensor<2, Int>,
        targets: Tensor<2, Int>,
        loss_scale: f64,
    ) -> PrecisionDiagnostics {
        let hidden = self.hidden(tokens);
        let activation = match self.head_dtype {
            Some(dtype) => hidden.clone().detach().cast(dtype).cast(FloatDType::F32),
            None => hidden.clone().detach(),
        };
        let activation_min = activation.clone().min().into_scalar::<f32>();
        let activation_max = activation.clone().max().into_scalar::<f32>();
        let activation_mean = activation.clone().mean().into_scalar::<f32>();
        let activation_nonfinite = nonfinite_count(activation);

        let logits = self.project(hidden);
        let logits_nonfinite = nonfinite_count(logits.clone().detach());
        let loss = Loss::new(logits, targets, self.z_loss);
        let total = loss.total.clone().detach().into_scalar::<f32>();
        let grads = loss.total.mul_scalar(loss_scale).backward();
        let embedding_grad =
            self.embed.weight.val().grad(&grads).expect("embedding must receive a gradient");
        let gradient_nonfinite = nonfinite_count(embedding_grad.clone());
        let embedding_grad = embedding_grad.div_scalar(loss_scale);
        let grad_norm = (embedding_grad.clone() * embedding_grad).sum().sqrt().into_scalar::<f32>();

        PrecisionDiagnostics {
            activation_min,
            activation_max,
            activation_mean,
            grad_norm,
            loss_scale: loss_scale as f32,
            nonfinite_count: activation_nonfinite + logits_nonfinite + gradient_nonfinite,
            loss: total,
        }
    }

    /// Next-token loss over every position.
    pub fn loss(&self, tokens: Tensor<2, Int>, targets: Tensor<2, Int>) -> Loss {
        Loss::new(self.forward(tokens), targets, self.z_loss)
    }
}

/// Scalars required by the precision quality gate.
#[derive(Clone, Copy, Debug)]
pub struct PrecisionDiagnostics {
    pub activation_min: f32,
    pub activation_max: f32,
    pub activation_mean: f32,
    pub grad_norm: f32,
    pub loss_scale: f32,
    pub nonfinite_count: usize,
    pub loss: f32,
}

fn nonfinite_count<const D: usize>(tensor: Tensor<D>) -> usize {
    tensor.is_finite().bool_not().int().sum().into_scalar::<i64>() as usize
}

fn tied_head(hidden: Tensor<3>, embedding: Tensor<2>, dtype: Option<FloatDType>) -> Tensor<3> {
    match dtype {
        Some(dtype) => hidden
            .cast(dtype)
            .matmul(embedding.cast(dtype).transpose().unsqueeze())
            .cast(FloatDType::F32),
        None => hidden.matmul(embedding.transpose().unsqueeze()),
    }
}

/// The training objective and its parts, kept separate so a run can log the
/// cross-entropy that perplexity is computed from rather than the total.
#[derive(Debug)]
pub struct Loss {
    pub nll: Tensor<1>,
    pub z: Tensor<1>,
    pub total: Tensor<1>,
}

impl Loss {
    fn new(logits: Tensor<3>, targets: Tensor<2, Int>, z_loss: f64) -> Self {
        let [batch, seq, _] = logits.dims();
        let logp = log_softmax(logits.clone(), 2);
        let nll = -logp.clone().gather(2, targets.reshape([batch, seq, 1])).mean();

        // `log_softmax` subtracts the log-normaliser, so recovering it needs no
        // second reduction: any vocabulary column of `logits - logp` is it.
        let column = [0..batch, 0..seq, 0..1];
        let z = (logits.slice(column.clone()) - logp.slice(column)).powi_scalar(2).mean();

        let total = nll.clone() + z.clone().mul_scalar(z_loss);
        Self { nll, z, total }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{Optim, Run};

    #[test]
    fn analytic_budget_matches_the_real_module() {
        let cfg = config::Model::toy();

        let model = Quasar::new(&cfg, &Device::default());

        assert_eq!(model.num_params(), cfg.budget().total, "\n{}", cfg.budget());
    }

    #[test]
    fn tied_and_untied_stacks_agree_on_shape() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);

        let tied = Quasar::new(&cfg.clone().with_tied_embeddings(true), &device);
        let untied = Quasar::new(&cfg.clone().with_tied_embeddings(false), &device);

        assert_eq!(tied.forward(tokens.clone()).dims(), [2, 8, cfg.vocab_size]);
        assert_eq!(untied.forward(tokens).dims(), [2, 8, cfg.vocab_size]);
    }

    #[test]
    fn f16_tied_head_keeps_fp32_masters_and_logits() {
        let cfg = config::Model::toy().with_tied_embeddings(true);
        let device = Device::default().autodiff();
        let reference = Quasar::new(&cfg, &device);
        let mixed = Quasar::new_with_head_dtype(&cfg, FloatDType::F16, &device)
            .load_record(reference.clone().into_record());
        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);
        assert_eq!(mixed.head_dtype, Some(FloatDType::F16));
        assert_eq!(mixed.embed.weight.val().dtype(), FloatDType::F32.into());

        let mixed_logits = mixed.forward(tokens);
        assert_eq!(mixed_logits.dtype(), FloatDType::F32.into());
    }

    #[test]
    fn f16_tied_head_matches_fp32_values_and_gradients_to_one_e_minus_two() {
        let device = Device::default().autodiff();
        let hidden_values: Vec<f32> = (0..32).map(|i| ((i % 13) as f32 - 6.0) / 7.0).collect();
        let embedding_values: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.0) / 11.0).collect();
        let hidden_data = TensorData::new(hidden_values, [1, 4, 8]);
        let embedding_data = TensorData::new(embedding_values, [16, 8]);

        let reference_hidden = Tensor::<3>::from_data(hidden_data.clone(), &device).require_grad();
        let reference_embedding =
            Tensor::<2>::from_data(embedding_data.clone(), &device).require_grad();
        let mixed_hidden = Tensor::<3>::from_data(hidden_data, &device).require_grad();
        let mixed_embedding = Tensor::<2>::from_data(embedding_data, &device).require_grad();

        let reference_logits = tied_head(reference_hidden, reference_embedding.clone(), None);
        let mixed_logits = tied_head(mixed_hidden, mixed_embedding.clone(), Some(FloatDType::F16));
        let logits_error =
            (reference_logits.clone() - mixed_logits.clone()).abs().max().into_scalar::<f32>();

        let reference_grads = reference_logits.powi_scalar(2).mean().backward();
        let mixed_grads = mixed_logits.powi_scalar(2).mean().mul_scalar(1024.0).backward();
        let reference_grad =
            reference_embedding.grad(&reference_grads).expect("reference embedding grad");
        let mixed_grad =
            mixed_embedding.grad(&mixed_grads).expect("mixed embedding grad").div_scalar(1024.0);
        let grad_error = (reference_grad - mixed_grad).abs().max().into_scalar::<f32>();

        assert!(logits_error > 0.0, "the reduced-precision seam was not exercised");
        assert!(logits_error < 1e-2, "maximum logits error {logits_error}");
        assert!(grad_error < 1e-2, "maximum embedding gradient error {grad_error}");
    }

    #[test]
    fn an_untrained_model_loses_about_log_vocab() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let tokens = Tensor::<2, Int>::zeros([2, 8], &device);

        let loss = Quasar::new(&cfg, &device).loss(tokens.clone(), tokens);

        let uniform = (cfg.vocab_size as f32).ln();
        assert!((loss.nll.into_scalar::<f32>() - uniform).abs() < 0.5, "expected ≈{uniform}");
    }

    #[test]
    fn a_later_token_sees_an_earlier_one() {
        let cfg = config::Model::toy();
        let device = Device::default();
        let base = Tensor::<2, Int>::zeros([1, 8], &device);
        let poked = base
            .clone()
            .slice_assign([0..1, 2..3], Tensor::<2, Int>::ones([1, 1], &device).mul_scalar(7));
        let model = Quasar::new(&cfg, &device);

        let delta = (model.forward(poked) - model.forward(base)).abs();

        let before = delta.clone().slice([0..1, 0..2, 0..cfg.vocab_size]).max();
        let after = delta.slice([0..1, 3..8, 0..cfg.vocab_size]).max();
        assert!(before.into_scalar::<f32>() < 1e-6, "position 2 must not leak backwards");
        assert!(after.into_scalar::<f32>() > 1e-6, "position 2 must reach later positions");
    }

    #[test]
    fn ssd_modes_agree_after_an_optimizer_step() {
        let cfg = config::Model::toy();
        let device = Device::default().autodiff().gradient_checkpointing();
        let record = Quasar::new(&cfg, &device).into_record();

        let step = |mode: config::SsdMode| -> f32 {
            let tokens = Tensor::<2, Int>::zeros([2, cfg.seq_len], &device);
            // Device seeding is global to the backend and races with other
            // tests running in parallel. Loading one shared record makes the
            // initial parameters identical and isolates the SSD algorithm.
            let mut model =
                Quasar::new_with_ssd(&cfg, mode.clone(), &device).load_record(record.clone());
            let run = Run::new().with_ssd_mode(Some(mode));
            let mut optim = Optim::new(&run, &model);

            let grads = model.loss(tokens.clone(), tokens.clone()).total.backward();
            optim.accumulate(&model, grads);
            model = optim.step(run.lr, model);

            model.loss(tokens.clone(), tokens).nll.into_scalar::<f32>()
        };

        let serial = step(config::SsdMode::Serial);
        let recalculated = step(config::SsdMode::Recalculated);
        let minimal = step(config::SsdMode::Minimal);

        assert!((serial - recalculated).abs() < 1e-4, "serial {serial} vs {recalculated}");
        assert!((minimal - recalculated).abs() < 1e-4, "minimal {minimal} vs {recalculated}");
    }
}
