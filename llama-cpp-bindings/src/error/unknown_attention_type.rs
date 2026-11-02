#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("attention type {0} is not a value llama.cpp defines")]
pub struct UnknownAttentionType(pub llama_cpp_bindings_sys::llama_attention_type);
