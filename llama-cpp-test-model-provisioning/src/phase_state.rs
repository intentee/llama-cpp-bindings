use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;

#[derive(Debug)]
pub struct PhaseState {
    pub mtmd_context: Option<MtmdContext>,
    pub model: LlamaModel,
    pub backend: Arc<LlamaBackend>,
    pub model_path: PathBuf,
}
