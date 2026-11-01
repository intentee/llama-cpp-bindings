use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings_tests::normalize::normalize;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const QUERY: &str = "What is machine learning?";
const RELEVANT_DOCUMENT: &str = "Machine learning is a subset of artificial intelligence.";
const UNRELATED_DOCUMENT: &str = "The weather today is sunny and warm.";

fn cosine_similarity(left_embedding: &[f32], right_embedding: &[f32]) -> f32 {
    normalize(left_embedding)
        .iter()
        .zip(normalize(right_embedding).iter())
        .map(|(left, right)| left * right)
        .sum::<f32>()
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 3,
    n_threads_batch = 8,
    embeddings = true,
)]
fn the_relevant_document_embeds_closer_to_the_query_than_an_unrelated_one(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    for (sequence_id, text) in (0..).zip([QUERY, RELEVANT_DOCUMENT, UNRELATED_DOCUMENT]) {
        let tokens =
            fixture
                .model
                .str_to_token(text, AddBos::Always, ParseSpecialTokens::Always)?;

        batch.add_sequence(&tokens, sequence_id, false)?;
    }

    context.decode(&batch)?;

    let query_embedding = context.embeddings_seq_ith(0)?;
    let relevant_similarity = cosine_similarity(query_embedding, context.embeddings_seq_ith(1)?);
    let unrelated_similarity = cosine_similarity(query_embedding, context.embeddings_seq_ith(2)?);

    assert!(
        relevant_similarity > unrelated_similarity,
        "the relevant document must rank above the unrelated one; \
         relevant={relevant_similarity} unrelated={unrelated_similarity}"
    );

    Ok(())
}
