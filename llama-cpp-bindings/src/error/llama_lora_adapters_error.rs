#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaLoraAdaptersError {
    #[error("llama_set_adapters_lora failed with status {0}")]
    ErrorResult(i32),
}
