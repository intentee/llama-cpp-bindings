#![cfg(all(feature = "llm-tests", feature = "llguidance"))]

use std::io::Write;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::{AddBos, LlamaModel};
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
fn json_schema_constrains_output() -> Result<()> {
    let model_path = download_model()?;

    let backend = LlamaBackend::init()?;
    let params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_path, &params)?;

    let prompt = "The weather in Paris is sunny and 22 degrees. Extract as JSON:\n";

    let ctx_params = LlamaContextParams::default();
    let mut ctx = model.new_context(&backend, ctx_params)?;

    let tokens_list = model.str_to_token(prompt, AddBos::Always)?;

    let mut batch = LlamaBatch::new(512, 1);
    let last_index = i32::try_from(tokens_list.len())? - 1;

    for (index, token) in (0_i32..).zip(&tokens_list) {
        batch.add(*token, index, &[0], index == last_index)?;
    }

    ctx.decode(&mut batch)?;

    let schema = r#"{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" }
  },
  "required": ["city", "temperature"]
}"#;

    let llg_sampler = LlamaSampler::llguidance(&model, "json", schema)?;
    let mut sampler = LlamaSampler::chain_simple([llg_sampler, LlamaSampler::greedy()]);

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut generated = String::new();

    while n_cur <= 128 {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);

        if model.is_eog_token(token) {
            break;
        }

        let output_string = model.token_to_piece(token, &mut decoder, true, None)?;
        generated.push_str(&output_string);
        print!("{output_string}");
        std::io::stdout().flush()?;

        sampler.accept(token)?;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    println!();

    let parsed: serde_json::Value = serde_json::from_str(&generated)?;

    assert!(
        parsed.get("city").is_some(),
        "constrained output should contain 'city' field"
    );
    assert!(
        parsed.get("temperature").is_some(),
        "constrained output should contain 'temperature' field"
    );

    Ok(())
}
