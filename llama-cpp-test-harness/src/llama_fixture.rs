use std::path::Path;
use std::sync::Arc;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;

use crate::context_params::ContextParams;

pub struct LlamaFixture<'fixture> {
    pub model: &'fixture Arc<LlamaModel>,
    pub backend: &'fixture LlamaBackend,
    pub context_params: &'fixture ContextParams,
    pub mtmd_context: Option<&'fixture MtmdContext>,
    pub model_path: &'fixture Path,
}

impl<'fixture> LlamaFixture<'fixture> {
    /// # Errors
    /// Forwards [`LlamaContext::from_model`] errors verbatim.
    pub fn build_context(&self) -> Result<LlamaContext<'fixture>, ContextError> {
        LlamaContext::from_model(
            self.model,
            self.backend,
            (*self.context_params).into_llama_context_params(),
        )
    }
}
