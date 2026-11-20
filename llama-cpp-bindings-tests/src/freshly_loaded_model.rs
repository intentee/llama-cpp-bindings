use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// Loads another copy of the fixture's model, which the caller owns and may change on its own.
///
/// # Errors
///
/// Forwards [`LlamaModel::load_from_file`] errors verbatim.
pub fn freshly_loaded_model(
    &LlamaFixture {
        backend,
        model_path,
        ..
    }: &LlamaFixture<'_>,
) -> Result<LlamaModel, ModelError> {
    LlamaModel::load_from_file(backend, model_path, &LlamaModelParams::default())
}
