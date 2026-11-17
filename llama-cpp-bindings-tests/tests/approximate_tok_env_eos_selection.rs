use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use std::path::Path;
use toktrie::INVALID_TOKEN;
use toktrie::TokenizerEnv as _;

use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const WORDPIECE_VOCABULARY_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../llama-cpp-gguf/fixtures/ggml-vocab-bert-bge.gguf"
);

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn approximate_tok_env_falls_back_to_eos_when_eot_unavailable(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let LlamaToken(eos) = model.token_eos().context("model declares an EOS token")?;

    ensure!(model.approximate_tok_env()?.tok_trie().eos_token() == eos.cast_unsigned());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn approximate_tok_env_of_a_vocabulary_without_eot_and_eos_has_no_eos(
    &LlamaFixture { backend, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let model_without_eos = LlamaModel::load_from_file(
        backend,
        Path::new(WORDPIECE_VOCABULARY_PATH),
        &LlamaModelParams::default().with_vocab_only(true),
    )?;

    ensure!(
        model_without_eos
            .approximate_tok_env()?
            .tok_trie()
            .eos_token()
            == INVALID_TOKEN
    );

    Ok(())
}
