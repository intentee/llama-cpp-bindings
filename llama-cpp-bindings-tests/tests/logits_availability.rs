use anyhow::Result;

use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::error::logits_error::LogitsError;
use llama_cpp_bindings::error::sample_error::SampleError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::data_array::LlamaTokenDataArray;
use llama_cpp_bindings_tests::context_after_decoding_text::context_after_decoding_text;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_pooled_embeddings_decode_produces_no_logits(fixture: &LlamaFixture<'_>) {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);

    assert_eq!(
        context.get_logits_ith(-1),
        Err(LogitsError::PooledEmbeddingsProduceNoLogits {
            pooling_type: LlamaPoolingType::Mean,
        })
    );
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
fn sampling_after_a_pooled_embeddings_decode_is_refused_instead_of_aborting(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);
    let mut sampler = LlamaSampler::greedy()?;

    assert_eq!(
        sampler.sample(&context, -1),
        Err(SampleError::Logits(
            LogitsError::PooledEmbeddingsProduceNoLogits {
                pooling_type: LlamaPoolingType::Mean,
            }
        ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("LiquidAI/LFM2-ColBERT-350M-GGUF", "LFM2-ColBERT-350M-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn a_model_whose_embeddings_graph_has_no_logits_head_produces_no_logits(
    fixture: &LlamaFixture<'_>,
) {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::None);

    assert_eq!(
        context.get_logits_ith(-1),
        Err(LogitsError::GraphProducesNoLogits)
    );
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
fn every_logits_accessor_refuses_a_pooled_embeddings_decode(fixture: &LlamaFixture<'_>) {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);

    assert_eq!(
        [
            context.get_logits_ith(-1).err(),
            context.candidates_ith(-1).err(),
            context.token_data_array_ith(-1).err(),
            context
                .fill_token_data_array_ith(-1, &mut LlamaTokenDataArray::new(Vec::new(), false))
                .err(),
        ],
        [(); 4].map(|()| {
            Some(LogitsError::PooledEmbeddingsProduceNoLogits {
                pooling_type: LlamaPoolingType::Mean,
            })
        })
    );
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
fn classified_sampling_after_a_pooled_embeddings_decode_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    assert_eq!(
        classifier
            .sample(&mut LlamaSampler::greedy()?, &context, -1, &mut Vec::new())
            .err(),
        Some(SampleError::Logits(
            LogitsError::PooledEmbeddingsProduceNoLogits {
                pooling_type: LlamaPoolingType::Mean,
            }
        ))
    );

    Ok(())
}
