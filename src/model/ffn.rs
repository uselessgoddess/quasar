//! The SwiGLU feed-forward sublayer.

use burn::nn::{Linear, LinearConfig, SwiGlu, SwiGluConfig};
use burn::prelude::*;
use burn::tensor::FloatDType;
use burn::tensor::activation::silu;
use burn::tensor::module::linear;

use crate::config;
use crate::model::init;

/// `down(silu(gate(x)) * up(x))` — three bias-free matrices, no dropout.
///
/// Dropout is absent on purpose: at 2–20 tokens per parameter the run is
/// nowhere near memorising the corpus, so the regulariser would only slow
/// convergence.
#[derive(Module, Debug)]
pub struct Ffn {
    gate: SwiGlu,
    down: Linear,
    /// Optional compute dtype for the three GEMMs only. Parameters, the SiLU
    /// and multiply, residual-stream inputs and outputs all remain fp32.
    #[module(skip)]
    dtype: Option<FloatDType>,
}

impl Ffn {
    pub fn new(cfg: &config::Model, device: &Device) -> Self {
        Self::new_with_optional_dtype(cfg, None, device)
    }

    /// Build an experiment that casts only the FFN GEMMs.
    pub fn new_with_dtype(cfg: &config::Model, dtype: FloatDType, device: &Device) -> Self {
        Self::new_with_optional_dtype(cfg, Some(dtype), device)
    }

    fn new_with_optional_dtype(
        cfg: &config::Model,
        dtype: Option<FloatDType>,
        device: &Device,
    ) -> Self {
        let d_ff = cfg.d_ff();
        Self {
            gate: SwiGluConfig::new(cfg.d_model, d_ff)
                .with_initializer(init::normal())
                .init(device),
            down: LinearConfig::new(d_ff, cfg.d_model)
                .with_bias(false)
                .with_initializer(init::residual(cfg.n_layers))
                .init(device),
            dtype,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let Some(dtype) = self.dtype else {
            return self.down.forward(self.gate.forward(x));
        };
        let x = x.cast(dtype);
        let inner = project(&self.gate.linear_inner, x.clone(), dtype);
        let outer = project(&self.gate.linear_outer, x, dtype);
        project(&self.down, silu(inner) * outer, dtype)
    }
}

fn project(layer: &Linear, x: Tensor<3>, dtype: FloatDType) -> Tensor<3> {
    linear(
        x.cast(dtype),
        layer.weight.val().cast(dtype),
        layer.bias.as_ref().map(|bias| bias.val().cast(dtype)),
    )
    .cast(FloatDType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::FloatDType;

    #[test]
    fn f16_gemms_keep_fp32_masters_and_match_values_and_gradients() {
        let cfg = config::Model::toy();
        let device = Device::default().autodiff();
        let reference = Ffn::new(&cfg, &device);
        let mixed = Ffn::new_with_dtype(&cfg, FloatDType::F16, &device)
            .load_record(reference.clone().into_record());
        let values: Vec<f32> =
            (0..(2 * cfg.seq_len * cfg.d_model)).map(|i| ((i % 29) as f32 - 14.0) / 17.0).collect();
        let data = TensorData::new(values, [2, cfg.seq_len, cfg.d_model]);
        let reference_input = Tensor::<3>::from_data(data.clone(), &device).require_grad();
        let mixed_input = Tensor::<3>::from_data(data, &device).require_grad();

        let reference_output = reference.forward(reference_input.clone());
        let mixed_output = mixed.forward(mixed_input.clone());
        let output_error =
            (reference_output.clone() - mixed_output.clone()).abs().max().into_scalar::<f32>();
        assert_eq!(mixed.gate.linear_inner.weight.val().dtype(), FloatDType::F32.into());
        assert_eq!(mixed_output.dtype(), FloatDType::F32.into());

        let reference_grads = reference_output.powi_scalar(2).mean().backward();
        let mixed_grads = mixed_output.powi_scalar(2).mean().mul_scalar(1024.0).backward();
        let reference_input_grad =
            reference_input.grad(&reference_grads).expect("reference input gradient");
        let mixed_input_grad =
            mixed_input.grad(&mixed_grads).expect("mixed input gradient").div_scalar(1024.0);
        let input_grad_error =
            (reference_input_grad - mixed_input_grad).abs().max().into_scalar::<f32>();
        let reference_weight_grad = reference
            .gate
            .linear_inner
            .weight
            .val()
            .grad(&reference_grads)
            .expect("reference weight gradient");
        let mixed_weight_grad = mixed
            .gate
            .linear_inner
            .weight
            .val()
            .grad(&mixed_grads)
            .expect("mixed weight gradient")
            .div_scalar(1024.0);
        let weight_grad_error =
            (reference_weight_grad - mixed_weight_grad).abs().max().into_scalar::<f32>();

        assert!(output_error > 0.0, "the reduced-precision seam was not exercised");
        assert!(output_error < 1e-2, "maximum output error {output_error}");
        assert!(input_grad_error < 1e-2, "maximum input gradient error {input_grad_error}");
        assert!(weight_grad_error < 1e-2, "maximum weight gradient error {weight_grad_error}");
    }
}
