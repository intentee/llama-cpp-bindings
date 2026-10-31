use anyhow::Result;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::batch_transfer_error::BatchTransferError;
use llama_cpp_bindings::error::encode_error::EncodeError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn encoding_more_tokens_than_the_micro_batch_returns_an_error(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture.model.str_to_token(
        &"hello ".repeat(100),
        AddBos::Never,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    let n_tokens = batch.n_tokens();

    assert_eq!(
        context.encode(&batch),
        Err(EncodeError::BatchExceedsMicroBatch {
            n_tokens,
            n_ubatch: context.n_ubatch(),
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn encoding_reports_a_token_llama_cpp_cannot_add_to_its_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = fixture.build_context()?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Never, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_BATCH_EXT_ASSIGN_ENTRIES_THROWS,
    );

    assert_eq!(
        context.encode(&batch),
        Err(EncodeError::BatchTransfer(BatchTransferError::Reported {
            message: injected_fault_message(),
        }))
    );

    Ok(())
}
