#![cfg(feature = "llm-tests")]

use std::time::Duration;

use anyhow::{Context, Result, bail};
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

fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>()
}

#[test]
fn reranking_produces_scores() -> Result<()> {
    let backend = LlamaBackend::init()?;
    let model_params = LlamaModelParams::default();
    let model_path = download_model()?;
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let query = "What is machine learning?";
    let documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny and warm.",
    ];

    let document_count = documents.len();

    let ctx_params = LlamaContextParams::default()
        .with_n_threads_batch(std::thread::available_parallelism()?.get().try_into()?)
        .with_n_seq_max(u32::try_from(document_count)?)
        .with_embeddings(true);
    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create context")?;

    let prompt_lines: Vec<String> = documents
        .iter()
        .map(|document| format!("{query}</s><s>{document}"))
        .collect();

    let tokens_lines_list = prompt_lines
        .iter()
        .map(|line| model.str_to_token(line, AddBos::Always))
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| "failed to tokenize prompts")?;

    let n_ctx = usize::try_from(ctx.n_ctx())?;

    if tokens_lines_list.iter().any(|tokens| n_ctx < tokens.len()) {
        bail!("one of the provided prompts exceeds the size of the context window");
    }

    let mut batch = LlamaBatch::new(2048, i32::try_from(document_count)?);
    let t_main_start = ggml_time_us();

    for (sequence_index, tokens) in tokens_lines_list.iter().enumerate() {
        batch.add_sequence(tokens, i32::try_from(sequence_index)?, false)?;
    }

    ctx.clear_kv_cache();
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let mut embeddings = Vec::with_capacity(document_count);

    for sequence_index in 0..document_count {
        let raw_embedding = ctx
            .embeddings_seq_ith(i32::try_from(sequence_index)?)
            .with_context(|| "failed to get sequence embeddings")?;
        embeddings.push(normalize(raw_embedding));
    }

    let t_main_end = ggml_time_us();
    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    let duration = Duration::from_micros(u64::try_from(t_main_end - t_main_start)?);

    #[allow(clippy::cast_precision_loss)]
    let tokens_per_second = total_tokens as f32 / duration.as_secs_f32();

    eprintln!(
        "created embeddings for {total_tokens} tokens in {:.2} s, speed {tokens_per_second:.2} t/s",
        duration.as_secs_f32(),
    );

    assert_eq!(
        embeddings.len(),
        document_count,
        "should produce one embedding per document"
    );

    for (index, embedding) in embeddings.iter().enumerate() {
        assert!(
            !embedding.is_empty(),
            "embedding {index} should not be empty"
        );
    }

    let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);
    eprintln!("cosine similarity between document embeddings: {similarity:.4}");

    assert!(
        similarity.is_finite(),
        "cosine similarity should be a finite number"
    );

    Ok(())
}
