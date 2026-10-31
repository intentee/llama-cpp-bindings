use std::env;
use std::fs;
use std::process;

use anyhow::Context;
use anyhow::Result;
use toktrie::INVALID_TOKEN;
use toktrie::TokenizerEnv as _;

use llama_cpp_bindings::gguf_context::GgufContext;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn approximate_tok_env_falls_back_to_eos_when_eot_unavailable(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let LlamaToken(eos) = fixture
        .model
        .token_eos()
        .context("model declares an EOS token")?;

    assert_eq!(
        fixture.model.approximate_tok_env()?.tok_trie().eos_token(),
        eos.cast_unsigned()
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 64,
    embeddings = true,
)]
fn approximate_tok_env_of_a_model_without_eot_and_eos_has_no_eos(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model_without_eos_path = env::temp_dir().join(format!(
        "llama_cpp_bindings_{}_model_without_eos.gguf",
        process::id()
    ));
    let mut gguf_context = GgufContext::from_file(fixture.model_path)?;

    gguf_context.remove_key("tokenizer.ggml.eos_token_id")?;
    gguf_context.write_to_file(&model_without_eos_path)?;

    let model_without_eos = LlamaModel::load_from_file(
        fixture.backend,
        &model_without_eos_path,
        &LlamaModelParams::default(),
    )?;

    fs::remove_file(&model_without_eos_path)?;

    assert_eq!(
        model_without_eos
            .approximate_tok_env()?
            .tok_trie()
            .eos_token(),
        INVALID_TOKEN
    );

    Ok(())
}
