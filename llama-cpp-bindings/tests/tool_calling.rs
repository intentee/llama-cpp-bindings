#![cfg(feature = "tests_that_use_llms")]

use std::io::Write;
use std::num::NonZeroU32;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
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
fn tool_calling_generates_grammar_and_prompt() -> Result<()> {
    let model_path = download_model()?;

    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)?;

    let template = model
        .chat_template(None)
        .unwrap_or_else(|_| LlamaChatTemplate::new("chatml").expect("valid chat template"));

    let messages = vec![
        LlamaChatMessage::new("system".to_string(), "You are a tool caller.".to_string())?,
        LlamaChatMessage::new(
            "user".to_string(),
            "Get the weather in Paris and summarize it.".to_string(),
        )?,
    ];

    let tools_json = serde_json::json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Fetch current weather by city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": { "type": "string", "description": "City name." },
                        "unit": { "type": "string", "enum": ["c", "f"] }
                    },
                    "required": ["city"]
                }
            }
        }
    ])
    .to_string();

    let result = model.apply_chat_template_with_tools_oaicompat(
        &template,
        &messages,
        Some(tools_json.as_str()),
        None,
        true,
    )?;

    assert!(
        !result.prompt.is_empty(),
        "prompt should not be empty after applying chat template with tools"
    );

    let tokens = model.str_to_token(&result.prompt, AddBos::Always)?;
    let n_predict: i32 = 128;
    let n_ctx = model
        .n_ctx_train()
        .max(tokens.len() as u32 + n_predict as u32);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1)?;
    let last_index = tokens.len().saturating_sub(1) as i32;

    for (index, token) in (0_i32..).zip(tokens.into_iter()) {
        let is_last = index == last_index;
        batch.add(token, index, &[0], is_last)?;
    }

    ctx.decode(&mut batch)?;

    let mut n_cur = batch.n_tokens();
    let max_tokens = n_cur + n_predict;

    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let (grammar_sampler, preserved) = result.build_grammar_sampler(&model)?;
    let mut sampler = if let Some(grammar) = grammar_sampler {
        LlamaSampler::chain_simple([grammar, LlamaSampler::greedy()])
    } else {
        LlamaSampler::greedy()
    };

    let mut generated_text = String::new();
    let additional_stops = result.additional_stops.clone();

    while n_cur <= max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved.contains(&token);
        let output_string = model.token_to_piece(token, &mut decoder, decode_special, None)?;
        generated_text.push_str(&output_string);
        print!("{output_string}");
        std::io::stdout().flush()?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;

        if additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop))
        {
            break;
        }
    }

    for stop in &additional_stops {
        if !stop.is_empty() && generated_text.ends_with(stop) {
            let new_len = generated_text.len().saturating_sub(stop.len());
            generated_text.truncate(new_len);

            break;
        }
    }

    let parsed_json = result.parse_response_oaicompat(&generated_text, false)?;
    let parsed_value: serde_json::Value = serde_json::from_str(&parsed_json)?;

    eprintln!(
        "\nParsed message:\n{}",
        serde_json::to_string_pretty(&parsed_value)?
    );

    assert!(
        !generated_text.is_empty(),
        "tool calling should generate output"
    );

    Ok(())
}
