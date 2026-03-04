//! # Usage
//!
//! Minimal inference example using llama-cpp-2. Downloads a model from
//! HuggingFace on first run:
//!
//! ```console
//! cargo run --example usage -- "What is the meaning of life?"
//! ```

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::io::Write;

const HF_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const HF_MODEL: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

fn main() -> anyhow::Result<()> {
    let user_prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Hello! How are you?".to_string());

    let model_path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(HF_REPO.to_string())
        .get(HF_MODEL)?;

    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)?;
    let context_params = LlamaContextParams::default();
    let mut context = model.new_context(&backend, context_params)?;

    let chat_template = model.chat_template(None)?;
    let messages = vec![LlamaChatMessage::new("user".to_string(), user_prompt)?];
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

    while position <= max_tokens {
        let token = sampler.sample(&context, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let piece = model.token_to_piece(token, &mut decoder, true, None)?;
        print!("{piece}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, position, &[0], true)?;
        position += 1;

        context.decode(&mut batch)?;
    }

    println!();

    Ok(())
}
