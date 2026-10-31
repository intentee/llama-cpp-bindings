use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

/// # Errors
/// Forwards tokenization, batch assembly, and [`LlamaContext::decode`] errors verbatim.
pub fn prime_memory_with(
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
