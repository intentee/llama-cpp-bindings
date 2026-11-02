use anyhow::Result;

use llama_cpp_bindings::model::vocabulary_token::VocabularyToken;
use llama_cpp_bindings_tests::single_token::single_token;
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
fn the_vocabulary_enumerates_each_token_at_its_id_with_its_piece(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let hello = single_token(model, "hello");

    assert_eq!(
        model.tokens(false).nth(usize::try_from(hello.0)?),
        Some(Ok(VocabularyToken {
            token: hello,
            piece: "hello".to_owned()
        }))
    );

    Ok(())
}
