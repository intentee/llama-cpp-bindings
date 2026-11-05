#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("pooling type {0} is not a value llama.cpp defines")]
pub struct UnknownPoolingType(pub llama_cpp_bindings_sys::llama_pooling_type);
