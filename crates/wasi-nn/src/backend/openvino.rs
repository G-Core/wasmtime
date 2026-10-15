//! Implements a `wasi-nn` [`BackendInner`] using OpenVINO.

use super::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
    NamedTensor, read,
};
use crate::wit::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use openvino::{
    DeviceType, ElementType, InferenceError, RwPropertyKey, SetupError, Shape, Tensor as OvTensor,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub struct OpenvinoBackend {
    core: Option<openvino::Core>,
    /// Device properties applied before every `compile_model`; see
    /// [`OpenvinoBackend::with_properties`].
    properties: Vec<(String, String)>,
}
unsafe impl Send for OpenvinoBackend {}
unsafe impl Sync for OpenvinoBackend {}

impl OpenvinoBackend {
    /// Construct a backend that sets `properties` on the execution device
    /// before compiling each model.
    ///
    /// Keys and values are the plain strings OpenVINO documents for
    /// `ov::Core::set_property` (e.g. `NUM_STREAMS`, `INFERENCE_NUM_THREADS`,
    /// `ENABLE_CPU_PINNING`, `PERFORMANCE_HINT`), so an embedder can forward
    /// them from its own configuration without this crate knowing every key.
    ///
    /// Without any property the CPU plugin compiles with its `LATENCY` default:
    /// a single stream whose one execution graph is guarded by a mutex held for
    /// the whole inference, so concurrent `compute` calls on a shared graph are
    /// serialized, and each inference is spread over every core of a socket.
    /// A graph served from many threads wants e.g. `NUM_STREAMS` equal to the
    /// desired concurrency and `INFERENCE_NUM_THREADS` sized to match.
    pub fn with_properties<K, V>(properties: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            core: None,
            properties: properties
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }
}

impl BackendInner for OpenvinoBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Openvino
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 2 {
            return Err(BackendError::InvalidNumberOfBuilders(2, builders.len()));
        }
        // Construct the context if none is present; this is done lazily (i.e.
        // upon actually loading a model) because it may fail to find and load
        // the OpenVINO libraries. The laziness limits the extent of the error
        // only to wasi-nn users, not all WASI users.
        if self.core.is_none() {
            self.core.replace(openvino::Core::new()?);
        }
        // Read the guest array.
        let xml = builders[0];
        let weights = builders[1];

        // Construct a new tensor for the model weights.
        let shape = Shape::new(&[1, weights.len() as i64])?;
        let mut weights_tensor = OvTensor::new(ElementType::U8, &shape)?;
        let buffer = weights_tensor.get_raw_data_mut()?;
        buffer.copy_from_slice(&weights);

        // Construct OpenVINO graph structures: `model` contains the graph
        // structure, `compiled_model` can perform inference.
        let core = self
            .core
            .as_mut()
            .expect("openvino::Core was previously constructed");
        let model = core.read_model_from_buffer(&xml, Some(&weights_tensor))?;
        // Properties are per device and only affect models compiled after they
        // are set, so (re)apply them for this load's target right before
        // compiling. They are not inherited from an earlier load on another
        // device.
        let device: DeviceType<'static> = target.into();
        for (key, value) in &self.properties {
            tracing::debug!("openvino: set {key}={value} on {}", device.as_ref());
            core.set_property(&device, &RwPropertyKey::Other(key.clone().into()), value)?;
        }
        let compiled_model = core.compile_model(&model, device)?;
        let box_: Box<dyn BackendGraph> =
            Box::new(OpenvinoGraph(Arc::new(Mutex::new(compiled_model))));
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OpenvinoBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.xml"))?;
        let weights = read(&path.join("model.bin"))?;
        self.load(&[&model, &weights], target)
    }
}

struct OpenvinoGraph(Arc<Mutex<openvino::CompiledModel>>);

unsafe impl Send for OpenvinoGraph {}
unsafe impl Sync for OpenvinoGraph {}

impl BackendGraph for OpenvinoGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let mut compiled_model = self.0.lock().unwrap();
        let infer_request = compiled_model.create_infer_request()?;
        let box_: Box<dyn BackendExecutionContext> =
            Box::new(OpenvinoExecutionContext(infer_request, self.0.clone()));
        Ok(box_.into())
    }
}

struct OpenvinoExecutionContext(openvino::InferRequest, Arc<Mutex<openvino::CompiledModel>>);

impl BackendExecutionContext for OpenvinoExecutionContext {
    fn set_input(&mut self, id: Id, tensor: &Tensor) -> Result<(), BackendError> {
        // Construct the tensor.
        let precision = tensor.ty.into();
        let dimensions = tensor
            .dimensions
            .iter()
            .map(|&d| d as i64)
            .collect::<Vec<_>>();
        let shape = Shape::new(&dimensions)?;
        let mut new_tensor = OvTensor::new(precision, &shape)?;
        let buffer = new_tensor.get_raw_data_mut()?;
        buffer.copy_from_slice(&tensor.data);
        // Assign the tensor to the request.
        match id {
            Id::Index(i) => self.0.set_input_tensor_by_index(i as usize, &new_tensor)?,
            Id::Name(name) => self.0.set_tensor(&name, &new_tensor)?,
        };
        Ok(())
    }

    fn compute(
        &mut self,
        inputs: Option<Vec<NamedTensor>>,
    ) -> Result<Option<Vec<NamedTensor>>, BackendError> {
        match inputs {
            // WIT
            Some(inputs) => {
                // Process all named inputs
                for input in &inputs {
                    let precision = input.tensor.ty.into();
                    let dimensions = input
                        .tensor
                        .dimensions
                        .iter()
                        .map(|&d| d as i64)
                        .collect::<Vec<_>>();
                    let shape = Shape::new(&dimensions)?;
                    let mut new_tensor = OvTensor::new(precision, &shape)?;
                    let buffer = new_tensor.get_raw_data_mut()?;
                    buffer.copy_from_slice(&input.tensor.data);

                    self.0.set_tensor(&input.name, &new_tensor)?;
                }

                // Run inference
                self.0.infer()?;

                // Get all outputs
                let compiled_model = self.1.lock().unwrap();
                let output_count = compiled_model.get_output_size()?;

                let mut output_tensors = Vec::new();
                for i in 0..output_count {
                    let output_tensor = self.0.get_output_tensor_by_index(i)?;

                    let dimensions = output_tensor
                        .get_shape()?
                        .get_dimensions()
                        .iter()
                        .map(|&dim| dim as u32)
                        .collect::<Vec<u32>>();

                    let ty = output_tensor.get_element_type()?.try_into()?;
                    let data = output_tensor.get_raw_data()?.to_vec();

                    // Currently openvino backend returns output index only, not output tensor name
                    output_tensors.push(NamedTensor {
                        name: format!("{i}"),
                        tensor: Tensor {
                            dimensions,
                            ty,
                            data,
                        },
                    });
                }
                Ok(Some(output_tensors))
            }

            // WITX
            None => {
                self.0.infer()?;
                Ok(None)
            }
        }
    }

    fn get_output(&mut self, id: Id) -> Result<Tensor, BackendError> {
        let output_name = match id {
            Id::Index(i) => self.0.get_output_tensor_by_index(i as usize)?,
            Id::Name(name) => self.0.get_tensor(&name)?,
        };
        let dimensions = output_name
            .get_shape()?
            .get_dimensions()
            .iter()
            .map(|&dim| dim as u32)
            .collect::<Vec<u32>>();
        let ty = output_name.get_element_type()?.try_into()?;
        let data = output_name.get_raw_data()?.to_vec();
        Ok(Tensor {
            dimensions,
            ty,
            data,
        })
    }
}

impl From<InferenceError> for BackendError {
    fn from(e: InferenceError) -> Self {
        BackendError::BackendAccess(wasmtime::Error::new(e))
    }
}

impl From<SetupError> for BackendError {
    fn from(e: SetupError) -> Self {
        BackendError::BackendAccess(wasmtime::Error::new(e))
    }
}

/// Return the execution target string expected by OpenVINO from the
/// `ExecutionTarget` enum provided by wasi-nn.
impl From<ExecutionTarget> for DeviceType<'static> {
    fn from(target: ExecutionTarget) -> Self {
        match target {
            ExecutionTarget::Cpu => DeviceType::CPU,
            ExecutionTarget::Gpu => DeviceType::GPU,
            ExecutionTarget::Tpu => {
                unimplemented!("OpenVINO does not support TPU execution targets")
            }
        }
    }
}

/// Return OpenVINO's precision type for the `TensorType` enum provided by
/// wasi-nn.
impl From<TensorType> for ElementType {
    fn from(tensor_type: TensorType) -> Self {
        match tensor_type {
            TensorType::Fp16 => ElementType::F16,
            TensorType::Fp32 => ElementType::F32,
            TensorType::Fp64 => ElementType::F64,
            TensorType::U8 => ElementType::U8,
            TensorType::I32 => ElementType::I32,
            TensorType::I64 => ElementType::I64,
            TensorType::Bf16 => ElementType::Bf16,
        }
    }
}

/// Return the `TensorType` enum provided by wasi-nn for OpenVINO's precision type
impl TryFrom<ElementType> for TensorType {
    type Error = BackendError;
    fn try_from(element_type: ElementType) -> Result<Self, Self::Error> {
        match element_type {
            ElementType::F16 => Ok(TensorType::Fp16),
            ElementType::F32 => Ok(TensorType::Fp32),
            ElementType::F64 => Ok(TensorType::Fp64),
            ElementType::U8 => Ok(TensorType::U8),
            ElementType::I32 => Ok(TensorType::I32),
            ElementType::I64 => Ok(TensorType::I64),
            ElementType::Bf16 => Ok(TensorType::Bf16),
            _ => Err(BackendError::UnsupportedTensorType(
                element_type.to_string(),
            )),
        }
    }
}
