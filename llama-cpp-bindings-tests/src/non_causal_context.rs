use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_params::llama_attention_type::LlamaAttentionType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// The fixture's context, created to attend non-causally.
///
/// # Errors
/// Forwards [`LlamaContext::from_model`] errors verbatim.
pub fn non_causal_context<'model>(
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
            .with_attention_type(LlamaAttentionType::NonCausal),
    )
}
