use llama_cpp_bindings::TokenAttrError;
use llama_cpp_bindings::TokenOutsideVocabulary;
use llama_cpp_bindings::TokenToStringError;
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
fn token_to_piece_bytes_rejects_tokens_outside_the_vocabulary(fixture: &LlamaFixture<'_>) {
    let model = fixture.model;
    let n_vocab = model.n_vocab();

    assert_eq!(
        [LlamaToken::new(-1), LlamaToken::new(n_vocab)]
            .map(|token| model.token_to_piece_bytes(token, 8, true, None)),
        [
            Err(TokenToStringError::TokenOutsideVocabulary(
                TokenOutsideVocabulary {
                    token: LlamaToken::new(-1),
                    n_vocab,
                }
            )),
            Err(TokenToStringError::TokenOutsideVocabulary(
                TokenOutsideVocabulary {
                    token: LlamaToken::new(n_vocab),
                    n_vocab,
                }
            )),
        ]
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn token_attr_rejects_tokens_outside_the_vocabulary(fixture: &LlamaFixture<'_>) {
    let model = fixture.model;
    let n_vocab = model.n_vocab();

    assert_eq!(
        model.token_attr(LlamaToken::new(n_vocab)).err(),
        Some(TokenAttrError::TokenOutsideVocabulary(
            TokenOutsideVocabulary {
                token: LlamaToken::new(n_vocab),
                n_vocab,
            }
        ))
    );
}
