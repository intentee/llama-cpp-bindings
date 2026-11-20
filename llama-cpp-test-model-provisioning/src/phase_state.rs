use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_model::llama_model::LlamaModel;

#[derive(Debug)]
pub struct PhaseState {
    pub model: Arc<LlamaModel>,
    pub backend: Arc<LlamaBackend>,
    pub model_path: PathBuf,
}
