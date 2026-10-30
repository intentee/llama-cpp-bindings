use anyhow::Result;
use llama_cpp_bindings::EncodeError;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
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
