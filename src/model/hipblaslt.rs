//! Autodiff-safe tied output head backed by hipBLASLt.

use burn::backend::autodiff::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind},
};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, BackendTypes, Dispatch, Rocm, Shape, backend_extension};
use burn::tensor::{DType, Tensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use burn_fusion::{
    Fusion, FusionBackend, FusionRuntime,
    stream::{Operation, StreamId},
};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
use cubecl_hip::HipRuntime;
use std::ffi::{CStr, c_char, c_void};
use std::marker::PhantomData;

unsafe extern "C" {
    fn quasar_hipblaslt_tied_head_forward(
        hidden: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i64,
        hidden_size: i64,
        vocab_size: i64,
    ) -> i32;
    fn quasar_hipblaslt_tied_head_backward(
        hidden: *const c_void,
        weight: *const c_void,
        output_grad: *const c_void,
        hidden_grad: *mut c_void,
        weight_grad: *mut c_void,
        rows: i64,
        hidden_size: i64,
        vocab_size: i64,
    ) -> i32;
    fn quasar_hipblaslt_last_error() -> *const c_char;
}

/// The two native operations are separate so autodiff retains only the tensors
/// needed for the exact tied-head VJP.
#[backend_extension(Rocm, Autodiff)]
trait HipblasltBackendExt: Backend {
    fn hipblaslt_tied_head(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    fn hipblaslt_tied_head_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let _ = (hidden, weight, output_grad);
        panic!("hipBLASLt backward is only invoked by the custom autodiff node")
    }
}

/// `[batch, seq, hidden] @ [vocab, hidden]^T`.
pub fn tied_head(hidden: Tensor<3>, weight: Tensor<2>) -> Tensor<3> {
    Tensor::from_dispatch(<Dispatch as HipblasltBackendExt>::hipblaslt_tied_head(
        hidden.into_dispatch(),
        weight.into_dispatch(),
    ))
}

fn native_error(status: i32) {
    if status == 0 {
        return;
    }
    // SAFETY: the C++ wrapper returns a thread-local, NUL-terminated string
    // whose storage remains alive until the next wrapper invocation.
    let message = unsafe {
        let pointer = quasar_hipblaslt_last_error();
        if pointer.is_null() {
            "unknown hipBLASLt error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    panic!("hipBLASLt tied head failed: {message}");
}

fn shape(
    hidden: &CubeTensor<HipRuntime>,
    weight: &CubeTensor<HipRuntime>,
) -> (usize, usize, usize) {
    assert_eq!(hidden.dtype, DType::F16, "hipBLASLt hidden must be f16");
    assert_eq!(weight.dtype, DType::F16, "hipBLASLt weight must be f16");
    assert_eq!(weight.meta.num_dims(), 2, "tied-head weight must be rank two");
    assert!(hidden.meta.num_dims() >= 2, "tied-head hidden tensor must have a matrix suffix");
    hidden.assert_is_on_same_device(weight);

    let hidden_dims = hidden.meta.shape();
    let weight_dims = weight.meta.shape();
    let hidden_size = hidden_dims[hidden_dims.len() - 1];
    assert_eq!(hidden_size, weight_dims[1], "tied-head hidden and weight dimensions must agree");
    let rows = hidden_dims[..hidden_dims.len() - 1].iter().product();
    (rows, hidden_size, weight_dims[0])
}

fn empty_like_shape(template: &CubeTensor<HipRuntime>, shape: Shape) -> CubeTensor<HipRuntime> {
    let handle = template.client.empty(shape.num_elements() * template.dtype.size());
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        handle,
        template.dtype,
    )
}

impl HipblasltBackendExt for CubeBackend<HipRuntime> {
    fn hipblaslt_tied_head(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let hidden = into_contiguous(hidden);
        let weight = into_contiguous(weight);
        let (rows, hidden_size, vocab_size) = shape(&hidden, &weight);
        let mut output_shape = hidden.meta.shape().clone();
        *output_shape.last_mut().expect("hidden rank checked above") = vocab_size;
        let output = empty_like_shape(&hidden, output_shape);

        let hidden_resource = hidden
            .client
            .get_resource(hidden.handle.clone())
            .expect("hidden HIP resource must be available");
        let weight_resource = weight
            .client
            .get_resource(weight.handle.clone())
            .expect("weight HIP resource must be available");
        let output_resource = output
            .client
            .get_resource(output.handle.clone())
            .expect("output HIP resource must be available");
        // SAFETY: resources keep all three Burn allocations alive for the call.
        // The wrapper synchronizes CubeCL work before borrowing the pointers and
        // its private stream before returning the initialized output.
        let status = unsafe {
            quasar_hipblaslt_tied_head_forward(
                hidden_resource.resource().ptr.cast_const().cast(),
                weight_resource.resource().ptr.cast_const().cast(),
                output_resource.resource().ptr.cast(),
                rows as i64,
                hidden_size as i64,
                vocab_size as i64,
            )
        };
        native_error(status);
        output
    }

    fn hipblaslt_tied_head_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let hidden = into_contiguous(hidden);
        let weight = into_contiguous(weight);
        let output_grad = into_contiguous(output_grad);
        let (rows, hidden_size, vocab_size) = shape(&hidden, &weight);
        assert_eq!(output_grad.dtype, DType::F16, "hipBLASLt gradient must be f16");
        assert_eq!(
            output_grad.meta.shape().iter().product::<usize>(),
            rows * vocab_size,
            "tied-head output gradient shape must agree with forward"
        );
        let hidden_grad = empty_like_shape(&hidden, hidden.meta.shape().clone());
        let weight_grad = empty_like_shape(&weight, weight.meta.shape().clone());

        let hidden_resource = hidden
            .client
            .get_resource(hidden.handle.clone())
            .expect("hidden HIP resource must be available");
        let weight_resource = weight
            .client
            .get_resource(weight.handle.clone())
            .expect("weight HIP resource must be available");
        let output_grad_resource = output_grad
            .client
            .get_resource(output_grad.handle.clone())
            .expect("output-gradient HIP resource must be available");
        let hidden_grad_resource = hidden_grad
            .client
            .get_resource(hidden_grad.handle.clone())
            .expect("hidden-gradient HIP resource must be available");
        let weight_grad_resource = weight_grad
            .client
            .get_resource(weight_grad.handle.clone())
            .expect("weight-gradient HIP resource must be available");
        // SAFETY: the managed resources pin all input and output allocations
        // through the synchronized native call.
        let status = unsafe {
            quasar_hipblaslt_tied_head_backward(
                hidden_resource.resource().ptr.cast_const().cast(),
                weight_resource.resource().ptr.cast_const().cast(),
                output_grad_resource.resource().ptr.cast_const().cast(),
                hidden_grad_resource.resource().ptr.cast(),
                weight_grad_resource.resource().ptr.cast(),
                rows as i64,
                hidden_size as i64,
                vocab_size as i64,
            )
        };
        native_error(status);
        (hidden_grad, weight_grad)
    }
}

impl<B: Backend + HipblasltBackendExt, C: CheckpointStrategy> HipblasltBackendExt
    for Autodiff<B, C>
{
    fn hipblaslt_tied_head(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct TiedHeadBackward;

        #[derive(Clone, Debug)]
        struct State<B: Backend> {
            hidden: <B as BackendTypes>::FloatTensorPrimitive,
            weight: <B as BackendTypes>::FloatTensorPrimitive,
        }

        impl<B: Backend + HipblasltBackendExt> Backward<B, 2> for TiedHeadBackward {
            type State = State<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 2>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let [hidden_node, weight_node] = ops.parents;
                let output_grad = grads.consume::<B>(&ops.node);
                let (hidden_grad, weight_grad) = B::hipblaslt_tied_head_backward(
                    ops.state.hidden,
                    ops.state.weight,
                    output_grad,
                );
                if let Some(node) = hidden_node {
                    grads.register::<B>(node.id, hidden_grad);
                }
                if let Some(node) = weight_node {
                    grads.register::<B>(node.id, weight_grad);
                }
            }
        }

        match TiedHeadBackward
            .prepare::<C>([hidden.node.clone(), weight.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let output =
                    B::hipblaslt_tied_head(hidden.primitive.clone(), weight.primitive.clone());
                let state = State { hidden: hidden.primitive, weight: weight.primitive };
                prep.finish(state, output)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::hipblaslt_tied_head(hidden.primitive, weight.primitive))
            }
        }
    }
}

#[derive(Clone, Debug)]
struct TiedHeadForward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + HipblasltBackendExt> Operation<B::FusionRuntime> for TiedHeadForward<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([hidden, weight], [output]) = self.desc.as_fixed();
        let result = B::hipblaslt_tied_head(
            handles.get_float_tensor::<B>(hidden),
            handles.get_float_tensor::<B>(weight),
        );
        handles.register_float_tensor::<B>(&output.id, result);
    }
}

#[derive(Clone, Debug)]
struct TiedHeadBackward<B> {
    desc: CustomOpIr,
    backend: PhantomData<B>,
}

impl<B: FusionBackend + HipblasltBackendExt> Operation<B::FusionRuntime> for TiedHeadBackward<B> {
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([hidden, weight, output_grad], [hidden_grad, weight_grad]) = self.desc.as_fixed();
        let (hidden_result, weight_result) = B::hipblaslt_tied_head_backward(
            handles.get_float_tensor::<B>(hidden),
            handles.get_float_tensor::<B>(weight),
            handles.get_float_tensor::<B>(output_grad),
        );
        handles.register_float_tensor::<B>(&hidden_grad.id, hidden_result);
        handles.register_float_tensor::<B>(&weight_grad.id, weight_result);
    }
}

impl<B: FusionBackend + HipblasltBackendExt> HipblasltBackendExt for Fusion<B> {
    fn hipblaslt_tied_head(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let [batch, seq, _] = hidden.shape.dims::<3>();
        let [vocab, _] = weight.shape.dims::<2>();
        let client = hidden.client.clone();
        let output = TensorIr::uninit(
            client.create_empty_handle(),
            Shape::new([batch, seq, vocab]),
            hidden.dtype,
        );
        let desc = CustomOpIr::new(
            "quasar_hipblaslt_tied_head_forward",
            &[hidden.into_ir(), weight.into_ir()],
            &[output],
        );
        client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                TiedHeadForward::<B> { desc, backend: PhantomData },
            )
            .output()
    }

    fn hipblaslt_tied_head_backward(
        hidden: FloatTensor<Self>,
        weight: FloatTensor<Self>,
        output_grad: FloatTensor<Self>,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let client = hidden.client.clone();
        let hidden_grad =
            TensorIr::uninit(client.create_empty_handle(), hidden.shape.clone(), hidden.dtype);
        let weight_grad =
            TensorIr::uninit(client.create_empty_handle(), weight.shape.clone(), weight.dtype);
        let desc = CustomOpIr::new(
            "quasar_hipblaslt_tied_head_backward",
            &[hidden.into_ir(), weight.into_ir(), output_grad.into_ir()],
            &[hidden_grad, weight_grad],
        );
        let [hidden_grad, weight_grad] = client
            .register(
                StreamId::current(),
                OperationIr::Custom(desc.clone()),
                TiedHeadBackward::<B> { desc, backend: PhantomData },
            )
            .try_into()
            .expect("hipBLASLt tied-head backward registers two outputs");
        (hidden_grad, weight_grad)
    }
}

#[cfg(test)]
mod tests {
    use super::tied_head;
    use burn::prelude::*;
    use burn::tensor::FloatDType;

    #[test]
    fn native_tied_head_matches_rocm_forward_and_both_gradients() {
        let device = Device::default().autodiff();
        let hidden_values: Vec<f32> =
            (0..2 * 8 * 64).map(|i| ((i % 31) as f32 - 15.0) / 64.0).collect();
        let weight_values: Vec<f32> =
            (0..128 * 64).map(|i| ((i % 37) as f32 - 18.0) / 96.0).collect();
        let hidden_data = TensorData::new(hidden_values, [2, 8, 64]);
        let weight_data = TensorData::new(weight_values, [128, 64]);

        let reference_hidden = Tensor::<3>::from_data(hidden_data.clone(), &device)
            .cast(FloatDType::F16)
            .require_grad();
        let reference_weight = Tensor::<2>::from_data(weight_data.clone(), &device)
            .cast(FloatDType::F16)
            .require_grad();
        let native_hidden =
            Tensor::<3>::from_data(hidden_data, &device).cast(FloatDType::F16).require_grad();
        let native_weight =
            Tensor::<2>::from_data(weight_data, &device).cast(FloatDType::F16).require_grad();

        let reference_output =
            reference_hidden.clone().matmul(reference_weight.clone().transpose().unsqueeze());
        eprintln!("hipBLASLt regression: materializing reference ROCm matmul");
        let _ = reference_output.clone().into_data();
        eprintln!("hipBLASLt regression: reference ROCm matmul complete");
        let native_output = tied_head(native_hidden.clone(), native_weight.clone());
        eprintln!("hipBLASLt regression: materializing native forward");
        let _ = native_output.clone().into_data();
        eprintln!("hipBLASLt regression: native forward complete");
        eprintln!("hipBLASLt regression: comparing forward outputs");
        let output_error = (reference_output.clone().cast(FloatDType::F32)
            - native_output.clone().cast(FloatDType::F32))
        .abs()
        .max()
        .into_scalar::<f32>();
        eprintln!("hipBLASLt regression: forward output comparison complete");

        eprintln!("hipBLASLt regression: running reference backward");
        let reference_grads = reference_output.powi_scalar(2).mean().mul_scalar(1024.0).backward();
        eprintln!("hipBLASLt regression: running native backward");
        let native_grads = native_output.powi_scalar(2).mean().mul_scalar(1024.0).backward();
        let reference_hidden_grad = reference_hidden
            .grad(&reference_grads)
            .expect("reference hidden gradient")
            .div_scalar(1024.0);
        let reference_weight_grad = reference_weight
            .grad(&reference_grads)
            .expect("reference weight gradient")
            .div_scalar(1024.0);
        let native_hidden_grad =
            native_hidden.grad(&native_grads).expect("native hidden gradient").div_scalar(1024.0);
        let native_weight_grad =
            native_weight.grad(&native_grads).expect("native weight gradient").div_scalar(1024.0);
        let hidden_grad_error = (reference_hidden_grad.cast(FloatDType::F32)
            - native_hidden_grad.cast(FloatDType::F32))
        .abs()
        .max()
        .into_scalar::<f32>();
        let weight_grad_error = (reference_weight_grad.cast(FloatDType::F32)
            - native_weight_grad.cast(FloatDType::F32))
        .abs()
        .max()
        .into_scalar::<f32>();

        assert!(output_error.is_finite(), "non-finite output error");
        assert!(hidden_grad_error.is_finite(), "non-finite hidden-gradient error");
        assert!(weight_grad_error.is_finite(), "non-finite weight-gradient error");
        assert!(output_error <= 1e-2, "maximum output error {output_error}");
        assert!(hidden_grad_error <= 1e-2, "maximum hidden-gradient error {hidden_grad_error}");
        assert!(weight_grad_error <= 1e-2, "maximum weight-gradient error {weight_grad_error}");
    }
}
