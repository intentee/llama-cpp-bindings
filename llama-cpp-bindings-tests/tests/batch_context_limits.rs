use anyhow::Result;

use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::batch_transfer_error::BatchTransferError;
use llama_cpp_bindings::error::decode_error::DecodeError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        n_batch: context_limits.n_batch * 2,
        ..context_limits
    };
    let mut batch = LlamaBatch::new(other_limits);
    batch.add(&SampledToken::Content(LlamaToken::new(1)), 0, &[0], true)?;

    assert_eq!(
        context.decode(&batch),
        Err(DecodeError::BatchTransfer(
            BatchTransferError::LimitsMismatch {
                batch: other_limits,
                context: context_limits,
            }
        ))
    );

    Ok(())
}
