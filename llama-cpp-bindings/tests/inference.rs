#![cfg(feature = "llm-tests")]

use std::io::Write;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_bindings::sampling::LlamaSampler;

const HF_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const HF_MODEL: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

fn download_model() -> Result<std::path::PathBuf> {
    let path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(HF_REPO.to_string())
        .get(HF_MODEL)?;

    Ok(path)
}

#[test]
fn chat_inference_produces_coherent_output() -> Result<()> {
    let model_path = download_model()?;

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    let context_params = LlamaContextParams::default();
    let mut context = model.new_context(&backend, context_params)?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new(
        "user".to_string(),
        "Hello! How are you?".to_string(),
    )?];
    let prompt = model.apply_chat_template(&chat_template, &messages, true)?;

    let tokens = model.str_to_token(&prompt, AddBos::Always)?;
    let mut batch = LlamaBatch::new(512, 1);

    let last_index = i32::try_from(tokens.len())? - 1;
    for (position, token) in (0_i32..).zip(tokens.into_iter()) {
        let output_logits = position == last_index;
        batch.add(token, position, &[0], output_logits)?;
    }
    context.decode(&mut batch)?;

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::greedy();
    let mut position = batch.n_tokens();
    let max_tokens = 1024;
    let mut generated = String::new();

    while position <= max_tokens {
        let token = sampler.sample(&context, batch.n_tokens() - 1);
        sampler.accept(token)?;

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)?;
        generated.push_str(&piece);
        print!("{piece}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, position, &[0], true)?;
        position += 1;

        context.decode(&mut batch)?;
    }

    println!();

    assert!(
        !generated.is_empty(),
        "model should generate at least one token"
    );

    Ok(())
}
