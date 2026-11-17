use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::batch_transfer_error::BatchTransferError;
use llama_cpp_bindings::error::decode_error::DecodeError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn decoding_a_batch_built_for_other_context_limits_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let context_limits = ContextLimits::of(&context);
    let other_limits = ContextLimits {
        batch: context_limits
            .batch
            .checked_mul(2)
            .context("twice the batch size must fit u32")?,
        ..context_limits
    };
    let mut batch = LlamaBatch::new(other_limits);
    batch.add(SampledToken::Content(LlamaToken::new(1)), 0, &[0], true)?;

    ensure!(
        context.decode(&batch)
            == Err(DecodeError::BatchTransfer(
                BatchTransferError::LimitsMismatch {
                    batch: other_limits,
                    context: context_limits,
                }
            ))
    );

    Ok(())
}
