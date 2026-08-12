#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlamaLoadModeParseError {
    pub value: llama_cpp_bindings_sys::llama_load_mode,
    pub context: String,
}
