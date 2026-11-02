#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("rope scaling type {0} is not a value llama.cpp defines")]
pub struct UnknownRopeScalingType(pub llama_cpp_bindings_sys::llama_rope_scaling_type);
