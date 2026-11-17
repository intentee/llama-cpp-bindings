use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::model_error::ModelError;
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
fn the_text_of_a_token_is_the_vocabulary_entry_it_was_tokenized_from(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(model.token_text_bytes(single_token(model, "Hello")?)? == b"Hello");

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
fn the_text_of_a_token_outside_the_vocabulary_is_refused(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let n_vocab = model.n_vocab();

    assert_eq!(
        model.token_text_bytes(LlamaToken::new(n_vocab)),
        Err(ModelError::TokenOutsideVocabulary {
            token: LlamaToken::new(n_vocab),
            n_vocab,
        })
    );
}
