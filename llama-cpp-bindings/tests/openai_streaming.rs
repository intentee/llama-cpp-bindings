#![cfg(feature = "llm-tests")]

use std::num::NonZeroU32;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_bindings::sampling::LlamaSampler;
use serde_json::json;

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
fn streaming_deltas_produce_valid_chunks() -> Result<()> {
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

    let tools_json = json!([
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

    let mut stream_state = result.streaming_state_oaicompat()?;
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let completion_id = format!("chatcmpl-{created}");
    let model_name = "test-model";
    let mut generated_text = String::new();
    let additional_stops = result.additional_stops.clone();
    let mut total_chunks = 0usize;

    while n_cur <= max_tokens {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved.contains(&token);
        let output_string = model.token_to_piece(token, &mut decoder, decode_special, None)?;
        generated_text.push_str(&output_string);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;

        let stop_now = additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop));
        let deltas = stream_state.update(&output_string, !stop_now)?;

        for delta in deltas {
            let delta_value: serde_json::Value = serde_json::from_str(&delta)?;
            let chunk = json!({
                "choices": [{
                    "delta": delta_value,
                    "finish_reason": serde_json::Value::Null,
                    "index": 0
                }],
                "created": created,
                "id": completion_id,
                "model": model_name,
                "object": "chat.completion.chunk"
            });

            let chunk_str = serde_json::to_string(&chunk)?;
            assert!(!chunk_str.is_empty(), "chunk should be valid JSON");
            total_chunks += 1;
        }

        if stop_now {
            break;
        }
    }

    eprintln!("streamed {total_chunks} delta chunks");

    assert!(
        !generated_text.is_empty(),
        "streaming should generate output"
    );

    Ok(())
}
