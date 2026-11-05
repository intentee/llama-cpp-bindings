#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("rope type {0} is not a value llama.cpp defines")]
pub struct UnknownRopeType(pub llama_cpp_bindings_sys::llama_rope_type);
