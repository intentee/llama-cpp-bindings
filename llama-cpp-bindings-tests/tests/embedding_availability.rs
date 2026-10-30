use llama_cpp_bindings::EmbeddingsError;
use llama_cpp_bindings::context::params::LlamaPoolingType;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

use llama_cpp_bindings_tests::context_after_decoding_text::context_after_decoding_text;

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
    embeddings = true,
)]
fn sequence_embeddings_need_a_pooling_context(fixture: &LlamaFixture<'_>) {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::None);

    assert_eq!(
        context.embeddings_seq_ith(0),
        Err(EmbeddingsError::NonePoolType)
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
fn token_embeddings_need_a_context_without_pooling(fixture: &LlamaFixture<'_>) {
    let context = context_after_decoding_text(fixture, LlamaPoolingType::Mean);

    assert_eq!(
        context.embeddings_ith(0),
        Err(EmbeddingsError::TokenEmbeddingsRequireNonePooling {
            pooling_type: LlamaPoolingType::Mean,
        })
    );
}
