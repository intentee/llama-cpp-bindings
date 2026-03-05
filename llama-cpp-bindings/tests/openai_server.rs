#![cfg(feature = "tests_that_use_llms")]

use std::collections::HashSet;
use std::num::NonZeroU32;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{AddBos, LlamaChatTemplate, LlamaModel};
use llama_cpp_bindings::openai::OpenAIChatTemplateParams;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use serde_json::{Value, json};

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

fn run_chat_completion(
    backend: &LlamaBackend,
    model: &LlamaModel,
    template: &LlamaChatTemplate,
    messages_json: &str,
    max_tokens: u32,
) -> Result<Value> {
    let params = OpenAIChatTemplateParams {
        messages_json,
        tools_json: None,
        tool_choice: None,
        json_schema: None,
        grammar: None,
        reasoning_format: None,
        chat_template_kwargs: None,
        add_generation_prompt: true,
        use_jinja: true,
        parallel_tool_calls: false,
        enable_thinking: true,
        add_bos: false,
        add_eos: false,
        parse_tool_calls: false,
    };

    let result = model.apply_chat_template_oaicompat(template, &params)?;

    let tokens = model.str_to_token(&result.prompt, AddBos::Always)?;
    let tokens_len_u32 = u32::try_from(tokens.len())?;
    let n_ctx = model.n_ctx_train().max(tokens_len_u32 + max_tokens);
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_ctx);
    let mut ctx = model.new_context(backend, ctx_params)?;

    let mut batch = LlamaBatch::new(n_ctx as usize, 1)?;
    let last_index = i32::try_from(tokens.len().saturating_sub(1))?;

    for (index, token) in (0_i32..).zip(tokens.iter().copied()) {
        let is_last = index == last_index;
        batch.add(token, index, &[0], is_last)?;
    }

    ctx.decode(&mut batch)?;

    let mut n_cur = batch.n_tokens();
    let max_tokens_i32 = i32::try_from(max_tokens)?;
    let max_tokens_total = n_cur + max_tokens_i32;
    let mut generated_text = String::new();
    let mut completion_tokens = 0u32;
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    let mut preserved = HashSet::<LlamaToken>::new();
    for token_str in &result.preserved_tokens {
        let tokens = model.str_to_token(token_str, AddBos::Never)?;
        if tokens.len() == 1 {
            preserved.insert(tokens[0]);
        }
    }

    let mut sampler = LlamaSampler::greedy();

    while n_cur < max_tokens_total {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(token) {
            break;
        }

        let decode_special = preserved.contains(&token);
        let output_string = model.token_to_piece(token, &mut decoder, decode_special, None)?;
        generated_text.push_str(&output_string);
        completion_tokens += 1;

        if result
            .additional_stops
            .iter()
            .any(|stop| !stop.is_empty() && generated_text.ends_with(stop))
        {
            break;
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    let finish_reason = if n_cur >= max_tokens_total {
        "length"
    } else {
        "stop"
    };

    for stop in &result.additional_stops {
        if !stop.is_empty() && generated_text.ends_with(stop) {
            let new_len = generated_text.len().saturating_sub(stop.len());
            generated_text.truncate(new_len);

            break;
        }
    }

    let message_json = result.parse_response_oaicompat(&generated_text, false)?;
    let message_value: Value = serde_json::from_str(&message_json)?;

    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let response = json!({
        "id": format!("chatcmpl-{}", created),
        "object": "chat.completion",
        "created": created,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": message_value,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": tokens_len_u32,
            "completion_tokens": completion_tokens,
            "total_tokens": tokens_len_u32 + completion_tokens
        }
    });

    Ok(response)
}

#[test]
fn openai_chat_completion_returns_valid_response() -> Result<()> {
    let model_path = download_model()?;

    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)?;
    let template = model.chat_template(None)?;

    let messages_json = json!([
        {"role": "user", "content": "Say hello in one word."}
    ])
    .to_string();

    let response = run_chat_completion(&backend, &model, &template, &messages_json, 64)?;

    assert_eq!(
        response["object"].as_str(),
        Some("chat.completion"),
        "response object type should be chat.completion"
    );

    let choices = response["choices"]
        .as_array()
        .expect("choices should be an array");
    assert!(!choices.is_empty(), "should have at least one choice");

    let message = &choices[0]["message"];
    assert!(message.is_object(), "choice message should be an object");

    let usage = &response["usage"];
    assert!(
        usage["prompt_tokens"].as_u64().unwrap_or(0) > 0,
        "prompt_tokens should be positive"
    );
    assert!(
        usage["completion_tokens"].as_u64().unwrap_or(0) > 0,
        "completion_tokens should be positive"
    );

    eprintln!("response: {}", serde_json::to_string_pretty(&response)?);

    Ok(())
}
