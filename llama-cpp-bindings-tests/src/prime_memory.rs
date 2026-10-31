use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;

use crate::prime_memory_with::prime_memory_with;

/// # Errors
/// Forwards tokenization and [`LlamaContext::decode`] errors verbatim.
pub fn prime_memory(fixture: &LlamaFixture<'_>, context: &mut LlamaContext<'_>) -> Result<()> {
    prime_memory_with(fixture, context, "Hello world")
}
