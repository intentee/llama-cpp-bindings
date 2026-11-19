use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::context_error::ContextError;
use llama_cpp_context::context_limits::ContextLimits;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_params::llama_attention_type::LlamaAttentionType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decoding_an_empty_batch_is_refused_as_invalid(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;
    let batch = LlamaBatch::new(context.limits());

    ensure!(context.decode(&batch) == Err(ContextError::DecodeBatchInvalid));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
)]
fn decoding_more_tokens_than_the_micro_batch_with_non_causal_attention_is_refused(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_attention_type(LlamaAttentionType::NonCausal),
    )?;
    let tokens = model.str_to_token(
        &"hello ".repeat(100),
        AddBos::Always,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    let n_tokens = batch.n_tokens();

    ensure!(
        context.decode(&batch)
            == Err(ContextError::DecodeBatchExceedsMicroBatch {
                n_tokens,
                n_ubatch: context.n_ubatch(),
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn decoding_more_tokens_than_the_micro_batch_on_an_encoder_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture.model.str_to_token(
        &"hello ".repeat(100),
        AddBos::Always,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    let n_tokens = batch.n_tokens();

    ensure!(
        context.decode(&batch)
            == Err(ContextError::DecodeBatchExceedsMicroBatch {
                n_tokens,
                n_ubatch: context.n_ubatch(),
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decoding_reports_an_exception_thrown_while_transferring_the_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_BATCH_EXT_ASSIGN_ENTRIES_THROWS,
    );

    ensure!(
        context.decode(&batch)
            == Err(ContextError::DecodeBatchTransfer(Box::new(
                ContextError::BatchTransferReported {
                    message: injected_fault_message()?,
                }
            )))
    );

    Ok(())
}

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
    let context_limits = context.limits();
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
            == Err(ContextError::DecodeBatchTransfer(Box::new(
                ContextError::BatchLimitsMismatch {
                    batch: other_limits,
                    context: context_limits,
                }
            )))
    );

    Ok(())
}
