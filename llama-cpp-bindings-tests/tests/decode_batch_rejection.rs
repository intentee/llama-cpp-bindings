use anyhow::Result;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_attention_type::LlamaAttentionType;
use llama_cpp_bindings::error::batch_transfer_error::BatchTransferError;
use llama_cpp_bindings::error::decode_error::DecodeError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decode_empty_batch_returns_error(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let batch = LlamaBatch::new(ContextLimits::of(&context));

    let result = context.decode(&batch);

    assert!(result.is_err());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
)]
fn decoding_more_tokens_than_the_micro_batch_with_non_causal_attention_returns_an_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_attention_type(LlamaAttentionType::NonCausal),
    )?;
    let tokens = fixture.model.str_to_token(
        &"hello ".repeat(100),
        AddBos::Always,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    let n_tokens = batch.n_tokens();

    assert_eq!(
        context.decode(&batch),
        Err(DecodeError::BatchExceedsMicroBatch {
            n_tokens,
            n_ubatch: context.n_ubatch(),
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn decoding_more_tokens_than_the_micro_batch_on_an_encoder_returns_an_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture.model.str_to_token(
        &"hello ".repeat(100),
        AddBos::Always,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    let n_tokens = batch.n_tokens();

    assert_eq!(
        context.decode(&batch),
        Err(DecodeError::BatchExceedsMicroBatch {
            n_tokens,
            n_ubatch: context.n_ubatch(),
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn decoding_reports_a_token_llama_cpp_cannot_add_to_its_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_BATCH_EXT_ADD_ENTRY_THROWS);

    assert_eq!(
        context.decode(&batch),
        Err(DecodeError::BatchTransfer(BatchTransferError::Reported {
            message: injected_fault_message(),
        }))
    );

    Ok(())
}
