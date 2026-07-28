//! One residual layer: a token mixer, then a feed-forward.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::FloatDType;
use burn_mamba::mamba3::prelude::{Mamba3, Mamba3SsdPath};

use crate::config::{self, Mixer, SsdMode};
use crate::model::{Attention, Ffn};

/// Whatever mixes tokens in this layer.
#[derive(Module, Debug)]
pub enum Mix {
    Ssm(Mamba3),
    Attn(Attention),
}

/// Pre-norm mixer sublayer plus pre-norm feed-forward sublayer, both residual.
///
/// Same shape for SSM and attention layers, so the stack is a uniform `Vec` and
/// the hybrid schedule is data, not control flow.
#[derive(Module, Debug)]
pub struct Block {
    norm_mix: RmsNorm,
    mix: Mix,
    norm_ffn: RmsNorm,
    ffn: Ffn,
    /// Chunk length of the SSD scan, resolved once at build time so the forward
    /// never falls back to burn-mamba's `None` (which pads the sequence).
    ssd_chunk: usize,
    /// Whether SSD intermediates are retained or recomputed in the backward.
    /// It changes execution only, not parameters or checkpoint records.
    #[module(skip)]
    ssd_mode: SsdMode,
}

impl Block {
    pub fn new(cfg: &config::Model, layer: usize, ssd_mode: SsdMode, device: &Device) -> Self {
        Self::new_with_projection_dtypes(cfg, layer, ssd_mode, None, None, device)
    }

    /// Build a block with an optional reduced compute dtype for its FFN GEMMs.
    pub fn new_with_ffn_dtype(
        cfg: &config::Model,
        layer: usize,
        ssd_mode: SsdMode,
        ffn_dtype: Option<FloatDType>,
        device: &Device,
    ) -> Self {
        Self::new_with_projection_dtypes(cfg, layer, ssd_mode, ffn_dtype, None, device)
    }

    /// Build with independent FFN and Mamba projection execution choices.
    pub fn new_with_projection_dtypes(
        cfg: &config::Model,
        layer: usize,
        ssd_mode: SsdMode,
        ffn_dtype: Option<FloatDType>,
        mamba_dtype: Option<FloatDType>,
        device: &Device,
    ) -> Self {
        let mix = match cfg.mixer(layer) {
            Mixer::Ssm => {
                let mamba = cfg.mamba().init(device);
                Mix::Ssm(match mamba_dtype {
                    Some(dtype) => mamba.with_projection_dtype(dtype),
                    None => mamba,
                })
            }
            Mixer::Attention => Mix::Attn(Attention::new(cfg, device)),
        };
        let ffn = match ffn_dtype {
            Some(dtype) => Ffn::new_with_dtype(cfg, dtype, device),
            None => Ffn::new(cfg, device),
        };
        Self {
            norm_mix: RmsNormConfig::new(cfg.d_model).init(device),
            mix,
            norm_ffn: RmsNormConfig::new(cfg.d_model).init(device),
            ffn,
            ssd_chunk: cfg.ssd_chunk_len(),
            ssd_mode,
        }
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let mixed = match &self.mix {
            // `SerialRecalculated` recomputes the SSD intermediates in the
            // backward instead of storing them — the difference between fitting
            // 16 GB and not. The chunk length is passed explicitly: left unset,
            // burn-mamba picks √(state_rank · head_dim) rounded to 32, which for
            // the shipped presets does not divide `seq_len` and makes every SSM
            // layer pad its sequence with six `cat` allocations.
            Mix::Ssm(ssm) => {
                let path = match self.ssd_mode {
                    SsdMode::Minimal => Mamba3SsdPath::Minimal(Some(self.ssd_chunk)),
                    SsdMode::Serial => Mamba3SsdPath::Serial(Some(self.ssd_chunk)),
                    SsdMode::Recalculated => {
                        Mamba3SsdPath::SerialRecalculated(Some(self.ssd_chunk))
                    }
                };
                ssm.forward(self.norm_mix.forward(x.clone()), None, path).0
            }
            Mix::Attn(attn) => attn.forward(self.norm_mix.forward(x.clone())),
        };
        let x = x + mixed;
        let ffn = self.ffn.forward(self.norm_ffn.forward(x.clone()));
        x + ffn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_mamba_projections_keep_fp32_masters_and_match_values_and_gradients() {
        let cfg = config::Model::toy();
        let device = Device::default().autodiff();
        let reference = cfg.mamba().init(&device);
        let mixed = cfg
            .mamba()
            .init(&device)
            .load_record(reference.clone().into_record())
            .with_projection_dtype(FloatDType::F16);
        let values: Vec<f32> =
            (0..(2 * cfg.seq_len * cfg.d_model)).map(|i| ((i % 31) as f32 - 15.0) / 37.0).collect();
        let data = TensorData::new(values, [2, cfg.seq_len, cfg.d_model]);
        let reference_input = Tensor::<3>::from_data(data.clone(), &device).require_grad();
        let mixed_input = Tensor::<3>::from_data(data, &device).require_grad();
        let path = Mamba3SsdPath::Serial(Some(cfg.ssd_chunk_len()));

        let reference_output = reference.forward(reference_input.clone(), None, path.clone()).0;
        let mixed_output = mixed.forward(mixed_input.clone(), None, path).0;
        let output_error =
            (reference_output.clone() - mixed_output.clone()).abs().max().into_scalar::<f32>();
        assert_eq!(mixed.in_proj.weight.val().dtype(), FloatDType::F32.into());
        assert_eq!(mixed.out_proj.weight.val().dtype(), FloatDType::F32.into());
        assert_eq!(mixed_output.dtype(), FloatDType::F32.into());
        let step_data = TensorData::new(
            (0..(2 * cfg.d_model)).map(|i| ((i % 17) as f32 - 8.0) / 29.0).collect::<Vec<_>>(),
            [2, cfg.d_model],
        );
        let reference_step =
            reference.step(Tensor::<2>::from_data(step_data.clone(), &device), None).0;
        let mixed_step = mixed.step(Tensor::<2>::from_data(step_data, &device), None).0;
        let step_error = (reference_step - mixed_step.clone()).abs().max().into_scalar::<f32>();
        assert_eq!(mixed_step.dtype(), FloatDType::F32.into());

        let reference_grads = reference_output.powi_scalar(2).mean().backward();
        let mixed_grads = mixed_output.powi_scalar(2).mean().mul_scalar(1024.0).backward();
        let reference_input_grad =
            reference_input.grad(&reference_grads).expect("reference input gradient");
        let mixed_input_grad =
            mixed_input.grad(&mixed_grads).expect("mixed input gradient").div_scalar(1024.0);
        let input_grad_error =
            (reference_input_grad - mixed_input_grad).abs().max().into_scalar::<f32>();

        let reference_in_weight_grad = reference
            .in_proj
            .weight
            .val()
            .grad(&reference_grads)
            .expect("reference input-projection weight gradient");
        let mixed_in_weight_grad = mixed
            .in_proj
            .weight
            .val()
            .grad(&mixed_grads)
            .expect("mixed input-projection weight gradient")
            .div_scalar(1024.0);
        let in_weight_grad_error =
            (reference_in_weight_grad - mixed_in_weight_grad).abs().max().into_scalar::<f32>();

        let reference_out_weight_grad = reference
            .out_proj
            .weight
            .val()
            .grad(&reference_grads)
            .expect("reference output-projection weight gradient");
        let mixed_out_weight_grad = mixed
            .out_proj
            .weight
            .val()
            .grad(&mixed_grads)
            .expect("mixed output-projection weight gradient")
            .div_scalar(1024.0);
        let out_weight_grad_error =
            (reference_out_weight_grad - mixed_out_weight_grad).abs().max().into_scalar::<f32>();

        assert!(output_error > 0.0, "the reduced-precision seam was not exercised");
        assert!(output_error < 1e-2, "maximum output error {output_error}");
        assert!(step_error < 1e-2, "maximum recurrent-step output error {step_error}");
        assert!(input_grad_error < 1e-2, "maximum input gradient error {input_grad_error}");
        assert!(
            in_weight_grad_error < 1e-2,
            "maximum input-projection weight gradient error {in_weight_grad_error}"
        );
        assert!(
            out_weight_grad_error < 1e-2,
            "maximum output-projection weight gradient error {out_weight_grad_error}"
        );
    }
}
