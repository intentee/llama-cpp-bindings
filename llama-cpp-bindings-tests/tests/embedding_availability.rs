use anyhow::Result;
use llama_cpp_bindings::EmbeddingsError;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::params::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings_types::BatchCapacity;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const TEXT: &str = "The parcel arrived on the last working day of the week.";

fn context_after_decoding_the_first_sequence<'fixture>(
    fixture: &'fixture LlamaFixture<'_>,
    pooling_type: LlamaPoolingType,
) -> Result<LlamaContext<'fixture>> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(pooling_type),
    )?;
    let tokens = fixture
        .model
        .str_to_token(TEXT, AddBos::Always, ParseSpecialTokens::Never)?;
    let mut batch = LlamaBatch::new(BatchCapacity::try_from(512)?, 1);

    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&mut batch)?;

    Ok(context)
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn sequence_embeddings_need_a_pooling_context(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = context_after_decoding_the_first_sequence(fixture, LlamaPoolingType::None)?;

    assert_eq!(
        context.embeddings_seq_ith(0),
        Err(EmbeddingsError::NonePoolType)
    );

    Ok(())
}
