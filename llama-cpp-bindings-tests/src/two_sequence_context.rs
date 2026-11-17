use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::llama_context_load_error::LlamaContextLoadError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// # Errors
/// Forwards [`LlamaContext::from_model`] errors verbatim.
pub fn two_sequence_context<'model>(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'model>,
) -> Result<LlamaContext<'model>, LlamaContextLoadError> {
    LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_kv_unified(true)
            .with_n_seq_max(2),
    )
}
