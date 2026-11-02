#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("backend device type {0} is not a value ggml defines")]
pub struct UnknownBackendDeviceType(pub llama_cpp_bindings_sys::ggml_backend_dev_type);
