use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// A context of two sequences that each keep their own KV stream.
///
/// # Errors
/// Forwards [`LlamaContext::from_model`] errors verbatim.
pub fn two_stream_context<'model>(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'model>,
) -> Result<LlamaContext<'model>, ContextError> {
    LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_kv_unified(false)
            .with_n_seq_max(2),
    )
}
