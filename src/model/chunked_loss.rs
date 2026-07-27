//! Memory-bounded language-model cross-entropy with an analytic backward.
//!
//! The ordinary loss materializes `[batch, seq, vocab]` logits and retains the
//! softmax graph. This operation flattens positions and handles at most
//! [`DEFAULT_CHUNK`] rows at once. Its custom backward saves only hidden
//! activations, the head weight and targets, then recomputes each logits chunk.

#[cfg(feature = "cuda")]
use burn::backend::Cuda;
#[cfg(feature = "flex")]
use burn::backend::Flex;
#[cfg(feature = "ndarray")]
use burn::backend::NdArray;
#[cfg(feature = "rocm")]
use burn::backend::Rocm;
#[cfg(feature = "vulkan")]
use burn::backend::Vulkan;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::backend_extension;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, BackendTypes, Dispatch, Shape, Slice, TensorMetadata};
use burn::prelude::*;

/// Positions per recomputed head projection.
pub const DEFAULT_CHUNK: usize = 256;

fn column_zero() -> [Slice; 2] {
    [Slice::from(..), Slice::from(0..1)]
}

fn chunk_rows(start: usize, end: usize) -> [Slice; 1] {
    [Slice::from(start..end)]
}

/// Forward NLL and z-loss sums, returned as `[mean_nll, mean_log_z_squared]`.
fn forward<B: Backend>(
    hidden_bsd: FloatTensor<B>,
    weight_vd: FloatTensor<B>,
    targets_bs: IntTensor<B>,
    chunk_size: usize,
) -> FloatTensor<B> {
    let [batch, seq, width] = hidden_bsd.shape().dims::<3>();
    let [vocab, weight_width] = weight_vd.shape().dims::<2>();
    assert_eq!(width, weight_width, "head width must match hidden width");
    assert_eq!([batch, seq], targets_bs.shape().dims::<2>(), "one target is required per position");
    assert!(chunk_size > 0, "cross-entropy chunk must be non-zero");

    let positions = batch * seq;
    let hidden_pd = B::float_reshape(hidden_bsd, Shape::new([positions, width]));
    let targets_p1 = B::int_reshape(targets_bs, Shape::new([positions, 1]));
    let weight_dv = B::float_transpose(weight_vd);
    let device = hidden_pd.device();
    let dtype = hidden_pd.dtype().into();
    let mut nll_sum = B::float_zeros(Shape::new([1]), &device, dtype);
    let mut z_sum = B::float_zeros(Shape::new([1]), &device, dtype);

    for start in (0..positions).step_by(chunk_size) {
        let end = (start + chunk_size).min(positions);
        let rows = chunk_rows(start, end);
        let hidden_cd = B::float_slice(hidden_pd.clone(), &rows);
        let target_c1 = B::int_slice(targets_p1.clone(), &rows);
        let logits_cv = B::float_matmul(hidden_cd, weight_dv.clone());
        assert_eq!([end - start, vocab], logits_cv.shape().dims::<2>());
        let logp_cv = B::log_softmax(logits_cv.clone(), 1);

        let selected_c1 = B::float_gather(1, logp_cv.clone(), target_c1);
        nll_sum = B::float_sub(nll_sum, B::float_sum(selected_c1));

        // logsumexp(logits) == logits[j] - log_softmax(logits)[j].
        let logits_c1 = B::float_slice(logits_cv, &column_zero());
        let logp_c1 = B::float_slice(logp_cv, &column_zero());
        let log_z_c1 = B::float_sub(logits_c1, logp_c1);
        z_sum = B::float_add(z_sum, B::float_sum(B::float_mul(log_z_c1.clone(), log_z_c1)));
    }

    B::float_cat(
        vec![
            B::float_div_scalar(nll_sum, (positions as f64).into()),
            B::float_div_scalar(z_sum, (positions as f64).into()),
        ],
        0,
    )
}

/// Exact VJP of [`forward`], recomputing only one logits chunk at a time.
fn backward<B: Backend>(
    hidden_bsd: FloatTensor<B>,
    weight_vd: FloatTensor<B>,
    targets_bs: IntTensor<B>,
    d_loss: FloatTensor<B>,
    chunk_size: usize,
) -> (FloatTensor<B>, FloatTensor<B>) {
    let [batch, seq, width] = hidden_bsd.shape().dims::<3>();
    let [vocab, weight_width] = weight_vd.shape().dims::<2>();
    assert_eq!(width, weight_width, "head width must match hidden width");
    assert_eq!([2], d_loss.shape().dims::<1>());

    let positions = batch * seq;
    let hidden_pd = B::float_reshape(hidden_bsd, Shape::new([positions, width]));
    let targets_p1 = B::int_reshape(targets_bs, Shape::new([positions, 1]));
    let weight_dv = B::float_transpose(weight_vd.clone());
    let d_nll = B::float_div_scalar(
        B::float_slice(d_loss.clone(), &[Slice::from(0..1)]),
        (positions as f64).into(),
    );
    let d_z = B::float_div_scalar(
        B::float_slice(d_loss, &[Slice::from(1..2)]),
        (positions as f64).into(),
    );
    let d_nll_11 = B::float_reshape(d_nll, Shape::new([1, 1]));
    let d_z_11 = B::float_reshape(d_z, Shape::new([1, 1]));

    let mut d_hidden_chunks = Vec::with_capacity(positions.div_ceil(chunk_size));
    let mut d_weight = None;
    for start in (0..positions).step_by(chunk_size) {
        let end = (start + chunk_size).min(positions);
        let chunk = end - start;
        let rows = chunk_rows(start, end);
        let hidden_cd = B::float_slice(hidden_pd.clone(), &rows);
        let target_c1 = B::int_slice(targets_p1.clone(), &rows);
        let logits_cv = B::float_matmul(hidden_cd.clone(), weight_dv.clone());
        let logp_cv = B::log_softmax(logits_cv.clone(), 1);
        let probability_cv = B::float_exp(logp_cv.clone());
        let log_z_c1 = B::float_sub(
            B::float_slice(logits_cv, &column_zero()),
            B::float_slice(logp_cv, &column_zero()),
        );

        // d logits = d_nll/N * (softmax - onehot)
        //          + d_z/N * 2*logsumexp * softmax.
        let nll_c1 = B::float_expand(d_nll_11.clone(), Shape::new([chunk, 1]));
        let z_c1 = B::float_expand(d_z_11.clone(), Shape::new([chunk, 1]));
        let coefficient_c1 = B::float_add(
            nll_c1.clone(),
            B::float_mul(B::float_mul_scalar(z_c1, 2.0.into()), log_z_c1),
        );
        let coefficient_cv = B::float_expand(coefficient_c1, Shape::new([chunk, vocab]));
        let d_logits_cv = B::float_mul(probability_cv, coefficient_cv);
        let d_logits_cv = B::float_scatter_add(1, d_logits_cv, target_c1, B::float_neg(nll_c1));

        d_hidden_chunks.push(B::float_matmul(d_logits_cv.clone(), weight_vd.clone()));
        let contribution_vd = B::float_matmul(B::float_transpose(d_logits_cv), hidden_cd);
        d_weight = Some(match d_weight {
            Some(accumulated) => B::float_add(accumulated, contribution_vd),
            None => contribution_vd,
        });
    }

    let d_hidden_bsd =
        B::float_reshape(B::float_cat(d_hidden_chunks, 0), Shape::new([batch, seq, width]));
    (d_hidden_bsd, d_weight.expect("at least one cross-entropy chunk"))
}

/// Backend operation for the memory-bounded head projection and loss.
#[backend_extension(
    Vulkan: cfg(feature = "vulkan"),
    Wgpu: cfg(feature = "wgpu"),
    Rocm: cfg(feature = "rocm"),
    Cuda: cfg(feature = "cuda"),
    Flex: cfg(feature = "flex"),
    NdArray: cfg(feature = "ndarray"),
    Autodiff: cfg(any(feature = "flex", feature = "ndarray", feature = "gpu")),
)]
pub trait ChunkedLossBackendExt: Backend {
    /// Return `[mean_nll, mean_log_z_squared]`.
    fn quasar_chunked_cross_entropy(
        hidden_bsd: FloatTensor<Self>,
        weight_vd: FloatTensor<Self>,
        targets_bs: IntTensor<Self>,
        chunk_size: usize,
    ) -> FloatTensor<Self> {
        forward::<Self>(hidden_bsd, weight_vd, targets_bs, chunk_size)
    }

    /// Recompute the chunked loss VJP for hidden activations and head weight.
    fn quasar_chunked_cross_entropy_backward(
        hidden_bsd: FloatTensor<Self>,
        weight_vd: FloatTensor<Self>,
        targets_bs: IntTensor<Self>,
        d_loss: FloatTensor<Self>,
        chunk_size: usize,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        backward::<Self>(hidden_bsd, weight_vd, targets_bs, d_loss, chunk_size)
    }
}

#[cfg(feature = "flex")]
impl ChunkedLossBackendExt for burn::backend::Flex {}

#[cfg(feature = "ndarray")]
impl<F, I> ChunkedLossBackendExt for burn::backend::NdArray<F, I> {}

#[cfg(feature = "gpu")]
impl<R: burn_cubecl::CubeRuntime> ChunkedLossBackendExt for burn_cubecl::CubeBackend<R> {}

impl<B: Backend + ChunkedLossBackendExt, C: CheckpointStrategy> ChunkedLossBackendExt
    for Autodiff<B, C>
{
    fn quasar_chunked_cross_entropy(
        hidden_bsd: FloatTensor<Self>,
        weight_vd: FloatTensor<Self>,
        targets_bs: IntTensor<Self>,
        chunk_size: usize,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct ChunkedLossBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            hidden: <B as BackendTypes>::FloatTensorPrimitive,
            weight: <B as BackendTypes>::FloatTensorPrimitive,
            targets: <B as BackendTypes>::IntTensorPrimitive,
            chunk_size: usize,
        }

        impl<B: Backend + ChunkedLossBackendExt> Backward<B, 2> for ChunkedLossBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [node_hidden, node_weight] = ops.parents;
                let d_loss = grads.consume::<B>(&ops.node);
                let (d_hidden, d_weight) = B::quasar_chunked_cross_entropy_backward(
                    ops.state.hidden,
                    ops.state.weight,
                    ops.state.targets,
                    d_loss,
                    ops.state.chunk_size,
                );
                if let Some(node) = node_hidden {
                    grads.register::<B>(node.id, d_hidden);
                }
                if let Some(node) = node_weight {
                    grads.register::<B>(node.id, d_weight);
                }
            }
        }

        match ChunkedLossBackward
            .prepare::<C>([hidden_bsd.node.clone(), weight_vd.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let combined = B::quasar_chunked_cross_entropy(
                    hidden_bsd.primitive.clone(),
                    weight_vd.primitive.clone(),
                    targets_bs.clone(),
                    chunk_size,
                );
                let state = State {
                    hidden: hidden_bsd.primitive,
                    weight: weight_vd.primitive,
                    targets: targets_bs,
                    chunk_size,
                };
                prep.finish(state, combined)
            }
            OpsKind::UnTracked(prep) => {
                let combined = B::quasar_chunked_cross_entropy(
                    hidden_bsd.primitive,
                    weight_vd.primitive,
                    targets_bs,
                    chunk_size,
                );
                prep.finish(combined)
            }
        }
    }
}

/// Return NLL and z-loss without retaining full-vocabulary logits.
pub fn chunked_cross_entropy(
    hidden_bsd: Tensor<3>,
    weight_vd: Tensor<2>,
    targets_bs: Tensor<2, Int>,
    chunk_size: usize,
) -> (Tensor<1>, Tensor<1>) {
    let combined =
        Tensor::from_dispatch(<Dispatch as ChunkedLossBackendExt>::quasar_chunked_cross_entropy(
            hidden_bsd.into_dispatch(),
            weight_vd.into_dispatch(),
            targets_bs.into_dispatch(),
            chunk_size,
        ));
    (combined.clone().narrow(0, 0, 1), combined.narrow(0, 1, 1))
}

#[cfg(feature = "gpu")]
mod fusion {
    use super::ChunkedLossBackendExt;
    use burn::backend::Shape;
    use burn::backend::tensor::{FloatTensor, IntTensor};
    use burn_fusion::{
        Fusion, FusionBackend, FusionRuntime,
        stream::{Operation, StreamId},
    };
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
    use core::marker::PhantomData;

    #[derive(Clone, Debug)]
    struct Forward<B> {
        desc: CustomOpIr,
        chunk_size: usize,
        backend: PhantomData<B>,
    }

    impl<B: FusionBackend + ChunkedLossBackendExt> Operation<B::FusionRuntime> for Forward<B> {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let ([hidden, weight, targets], [output]) = self.desc.as_fixed();
            let result = B::quasar_chunked_cross_entropy(
                handles.get_float_tensor::<B>(hidden),
                handles.get_float_tensor::<B>(weight),
                handles.get_int_tensor::<B>(targets),
                self.chunk_size,
            );
            handles.register_float_tensor::<B>(&output.id, result);
        }
    }

    #[derive(Clone, Debug)]
    struct Backward<B> {
        desc: CustomOpIr,
        chunk_size: usize,
        backend: PhantomData<B>,
    }

    impl<B: FusionBackend + ChunkedLossBackendExt> Operation<B::FusionRuntime> for Backward<B> {
        fn execute(
            &self,
            handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
        ) {
            let ([hidden, weight, targets, d_loss], [d_hidden, d_weight]) = self.desc.as_fixed();
            let (hidden_result, weight_result) = super::backward::<B>(
                handles.get_float_tensor::<B>(hidden),
                handles.get_float_tensor::<B>(weight),
                handles.get_int_tensor::<B>(targets),
                handles.get_float_tensor::<B>(d_loss),
                self.chunk_size,
            );
            handles.register_float_tensor::<B>(&d_hidden.id, hidden_result);
            handles.register_float_tensor::<B>(&d_weight.id, weight_result);
        }
    }

    impl<B: FusionBackend + ChunkedLossBackendExt> ChunkedLossBackendExt for Fusion<B> {
        fn quasar_chunked_cross_entropy(
            hidden_bsd: FloatTensor<Self>,
            weight_vd: FloatTensor<Self>,
            targets_bs: IntTensor<Self>,
            chunk_size: usize,
        ) -> FloatTensor<Self> {
            let client = hidden_bsd.client.clone();
            let output =
                TensorIr::uninit(client.create_empty_handle(), Shape::new([2]), hidden_bsd.dtype);
            let desc = CustomOpIr::new(
                "quasar_chunked_cross_entropy",
                &[hidden_bsd.into_ir(), weight_vd.into_ir(), targets_bs.into_ir()],
                &[output],
            );
            client
                .register(
                    StreamId::current(),
                    OperationIr::Custom(desc.clone()),
                    Forward::<B> { desc, chunk_size, backend: PhantomData },
                )
                .output()
        }

        fn quasar_chunked_cross_entropy_backward(
            hidden_bsd: FloatTensor<Self>,
            weight_vd: FloatTensor<Self>,
            targets_bs: IntTensor<Self>,
            d_loss: FloatTensor<Self>,
            chunk_size: usize,
        ) -> (FloatTensor<Self>, FloatTensor<Self>) {
            let [batch, seq, width] = hidden_bsd.shape.dims::<3>();
            let [vocab, _] = weight_vd.shape.dims::<2>();
            let client = hidden_bsd.client.clone();
            let dtype = hidden_bsd.dtype;
            let d_hidden = TensorIr::uninit(
                client.create_empty_handle(),
                Shape::new([batch, seq, width]),
                dtype,
            );
            let d_weight =
                TensorIr::uninit(client.create_empty_handle(), Shape::new([vocab, width]), dtype);
            let desc = CustomOpIr::new(
                "quasar_chunked_cross_entropy_backward",
                &[
                    hidden_bsd.into_ir(),
                    weight_vd.into_ir(),
                    targets_bs.into_ir(),
                    d_loss.into_ir(),
                ],
                &[d_hidden, d_weight],
            );
            let [d_hidden, d_weight] = client
                .register(
                    StreamId::current(),
                    OperationIr::Custom(desc.clone()),
                    Backward::<B> { desc, chunk_size, backend: PhantomData },
                )
                .try_into()
                .expect("chunked cross-entropy backward registers two outputs");
            (d_hidden, d_weight)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::activation::log_softmax;

    const Z_LOSS: f64 = 1e-4;

    fn values() -> (TensorData, TensorData, TensorData) {
        (
            TensorData::new(vec![0.2f32, -0.3, 0.7, 0.1, -0.4, 0.5], [1, 3, 2]),
            TensorData::new(vec![0.1f32, 0.3, -0.2, 0.4, 0.5, -0.1, -0.3, -0.2], [4, 2]),
            TensorData::new(vec![0i32, 2, 3], [1, 3]),
        )
    }

    fn reference(
        hidden: Tensor<3>,
        weight: Tensor<2>,
        targets: Tensor<2, Int>,
    ) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
        let [batch, seq, _] = hidden.dims();
        let logits = hidden.matmul(weight.transpose().unsqueeze());
        let logp = log_softmax(logits.clone(), 2);
        let nll = -logp.clone().gather(2, targets.reshape([batch, seq, 1])).mean();
        let column = [0..batch, 0..seq, 0..1];
        let z = (logits.slice(column.clone()) - logp.slice(column)).powi_scalar(2).mean();
        let total = nll.clone() + z.clone().mul_scalar(Z_LOSS);
        (nll, z, total)
    }

    fn max_abs_diff(a: Tensor<3>, b: Tensor<3>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    fn max_abs_diff_2d(a: Tensor<2>, b: Tensor<2>) -> f32 {
        (a - b).abs().max().into_scalar()
    }

    #[test]
    fn chunked_values_match_materialized_cross_entropy_to_1e_5() {
        let device = Device::default().autodiff();
        let (hidden, weight, targets) = values();
        let hidden = Tensor::from_data(hidden, &device);
        let weight = Tensor::from_data(weight, &device);
        let targets = Tensor::from_data(targets, &device);

        let (reference_nll, reference_z, _) =
            reference(hidden.clone(), weight.clone(), targets.clone());
        let (chunked_nll, chunked_z) = chunked_cross_entropy(hidden, weight, targets, 2);

        let nll_delta = (reference_nll - chunked_nll).abs().into_scalar::<f32>();
        let z_delta = (reference_z - chunked_z).abs().into_scalar::<f32>();
        assert!(nll_delta < 1e-5, "NLL delta {nll_delta}");
        assert!(z_delta < 1e-5, "z-loss delta {z_delta}");
    }

    #[test]
    fn analytic_backward_matches_materialized_autodiff_to_1e_5() {
        let device = Device::default().autodiff();
        let (hidden_data, weight_data, targets_data) = values();

        let hidden_reference = Tensor::from_data(hidden_data.clone(), &device).require_grad();
        let weight_reference = Tensor::from_data(weight_data.clone(), &device).require_grad();
        let targets_reference = Tensor::from_data(targets_data.clone(), &device);
        let (_, _, reference_total) =
            reference(hidden_reference.clone(), weight_reference.clone(), targets_reference);
        let reference_grads = reference_total.backward();
        let reference_hidden =
            hidden_reference.grad(&reference_grads).expect("reference hidden gradient");
        let reference_weight =
            weight_reference.grad(&reference_grads).expect("reference weight gradient");

        let hidden_chunked = Tensor::from_data(hidden_data, &device).require_grad();
        let weight_chunked = Tensor::from_data(weight_data, &device).require_grad();
        let targets_chunked = Tensor::from_data(targets_data, &device);
        let (nll, z) = chunked_cross_entropy(
            hidden_chunked.clone(),
            weight_chunked.clone(),
            targets_chunked,
            2,
        );
        let chunked_grads = (nll + z.mul_scalar(Z_LOSS)).backward();
        let chunked_hidden = hidden_chunked.grad(&chunked_grads).expect("chunked hidden gradient");
        let chunked_weight = weight_chunked.grad(&chunked_grads).expect("chunked weight gradient");

        let hidden_delta = max_abs_diff(reference_hidden, chunked_hidden);
        let weight_delta = max_abs_diff_2d(reference_weight, chunked_weight);
        assert!(hidden_delta < 1e-5, "hidden gradient delta {hidden_delta}");
        assert!(weight_delta < 1e-5, "weight gradient delta {weight_delta}");
    }

    #[test]
    fn analytic_backward_passes_a_centered_finite_difference_gradcheck() {
        let device = Device::default().autodiff();
        let (hidden_data, weight_data, targets_data) = values();
        let hidden = Tensor::from_data(hidden_data.clone(), &device).require_grad();
        let weight = Tensor::from_data(weight_data.clone(), &device).require_grad();
        let targets = Tensor::from_data(targets_data.clone(), &device);
        let (nll, z) = chunked_cross_entropy(hidden.clone(), weight.clone(), targets, 2);
        let grads = (nll + z.mul_scalar(Z_LOSS)).backward();
        let analytic_hidden = hidden
            .grad(&grads)
            .expect("hidden gradient")
            .into_data()
            .iter::<f32>()
            .next()
            .expect("first hidden gradient");
        let analytic_weight = weight
            .grad(&grads)
            .expect("weight gradient")
            .into_data()
            .iter::<f32>()
            .next()
            .expect("first weight gradient");

        let scalar_loss = |hidden_values: TensorData, weight_values: TensorData| {
            let hidden = Tensor::from_data(hidden_values, &device);
            let weight = Tensor::from_data(weight_values, &device);
            let targets = Tensor::from_data(targets_data.clone(), &device);
            let (nll, z) = chunked_cross_entropy(hidden, weight, targets, 2);
            (nll + z.mul_scalar(Z_LOSS)).into_scalar::<f32>()
        };
        let epsilon = 1e-3f32;

        let mut hidden_plus = hidden_data.to_vec::<f32>().expect("hidden data");
        let mut hidden_minus = hidden_plus.clone();
        hidden_plus[0] += epsilon;
        hidden_minus[0] -= epsilon;
        let numeric_hidden =
            (scalar_loss(TensorData::new(hidden_plus, [1, 3, 2]), weight_data.clone())
                - scalar_loss(TensorData::new(hidden_minus, [1, 3, 2]), weight_data.clone()))
                / (2.0 * epsilon);

        let mut weight_plus = weight_data.to_vec::<f32>().expect("weight data");
        let mut weight_minus = weight_plus.clone();
        weight_plus[0] += epsilon;
        weight_minus[0] -= epsilon;
        let numeric_weight =
            (scalar_loss(hidden_data.clone(), TensorData::new(weight_plus, [4, 2]))
                - scalar_loss(hidden_data, TensorData::new(weight_minus, [4, 2])))
                / (2.0 * epsilon);

        assert!(
            (analytic_hidden - numeric_hidden).abs() < 2e-3,
            "hidden analytic {analytic_hidden} vs numeric {numeric_hidden}"
        );
        assert!(
            (analytic_weight - numeric_weight).abs() < 2e-3,
            "weight analytic {analytic_weight} vs numeric {numeric_weight}"
        );
    }
}
