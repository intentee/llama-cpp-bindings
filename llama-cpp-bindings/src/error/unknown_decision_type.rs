#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("decision type {0} is not a value the llama.cpp wrapper defines")]
pub struct UnknownDecisionType(pub llama_cpp_bindings_sys::llama_rs_decision_type);
