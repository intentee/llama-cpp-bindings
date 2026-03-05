#![cfg(feature = "llm-tests")]

use std::time::Duration;

use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

const HF_REPO: &str = "Qwen/Qwen3-Embedding-0.6B-GGUF";
const HF_MODEL: &str = "Qwen3-Embedding-0.6B-Q8_0.gguf";

fn download_model() -> Result<std::path::PathBuf> {
    let path = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()?
        .model(HF_REPO.to_string())
        .get(HF_MODEL)?;

    Ok(path)
}

fn normalize(input: &[f32]) -> Vec<f32> {
    let magnitude = input
        .iter()
        .fold(0.0, |accumulator, &value| value.mul_add(value, accumulator))
        .sqrt();

    input.iter().map(|&value| value / magnitude).collect()
}

#[test]
fn embedding_generation_produces_vectors() -> Result<()> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model_path = download_model()?;
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| format!("failed to tokenize {prompt}"))?;

    let n_ctx = usize::try_from(ctx.n_ctx())?;
    assert!(tokens.len() <= n_ctx, "prompt exceeds context window size");

    let t_main_start = ggml_time_us();

    let mut batch = LlamaBatch::new(n_ctx, 1);
    batch.add_sequence(&tokens, 0, false)?;

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "failed to get embeddings")?;
    let normalized = normalize(embedding);

    let t_main_end = ggml_time_us();
    let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

    eprintln!(
        "created embedding with {} dimensions in {:.2} s",
        normalized.len(),
        duration.as_secs_f32()
    );

    assert!(
        !normalized.is_empty(),
        "embedding should have at least one dimension"
    );

    let magnitude: f32 = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "normalized embedding magnitude should be approximately 1.0, got {magnitude}"
    );

    Ok(())
}
