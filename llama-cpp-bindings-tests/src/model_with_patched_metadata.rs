use std::env;
use std::fs;
use std::sync::Arc;

use anyhow::Result;
use llama_cpp_gguf::gguf_context::GgufContext;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// Loads a copy of the fixture's model whose metadata `patch` changed, written under `file_name`
/// in the temporary directory and removed once loaded.
///
/// # Errors
///
/// Fails when the copy cannot be patched, written or loaded.
pub fn model_with_patched_metadata<TPatch>(
    &LlamaFixture {
        backend,
        model_path,
        ..
    }: &LlamaFixture<'_>,
    file_name: &str,
    patch: TPatch,
) -> Result<Arc<LlamaModel>>
where
    TPatch: FnOnce(&mut GgufContext) -> Result<()>,
{
    let mut gguf = GgufContext::from_file(model_path)?;
    let patched_model_path = env::temp_dir().join(file_name);

    patch(&mut gguf)?;
    gguf.write_to_file(&patched_model_path)?;

    let patched_model =
        LlamaModel::load_from_file(backend, &patched_model_path, &LlamaModelParams::default());

    fs::remove_file(&patched_model_path)?;

    Ok(Arc::new(patched_model?))
}
