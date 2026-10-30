use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_test_harness::LlamaFixture;

/// # Errors
/// Forwards tokenization, batch assembly, and [`LlamaContext::decode`] errors verbatim.
pub fn prime_kv_cache_with(
    fixture: &LlamaFixture<'_>,
    context: &mut LlamaContext<'_>,
    text: &str,
) -> Result<()> {
    let tokens = fixture
        .model
        .str_to_token(text, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;
    Ok(())
}
