use anyhow::Result;
use anyhow::ensure;

use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const DEEPSEEK_V4_SLIDING_WINDOW: u32 = 128;
const GEMMA_3_SLIDING_WINDOW: u32 = 1024;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn context_reports_positive_context_and_batch_sizes(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = fixture.build_context()?;

    ensure!(context.n_ctx() > 0);
    ensure!(context.n_batch() > 0);
    ensure!(context.n_ubatch() > 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 512,
    n_seq_max = 4,
)]
fn n_ctx_seq_splits_the_context_between_sequences(fixture: &LlamaFixture<'_>) -> Result<()> {
    let context = fixture.build_context()?;

    ensure!(context.n_ctx() == 2048);
    ensure!(context.n_ctx_seq() == 512);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_sliding_window_attention_context_limits_positions_by_the_window_its_cache_masks_with(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = fixture.build_context()?;

    ensure!(context.limits().sliding_window == GEMMA_3_SLIDING_WINDOW);

    Ok(())
}

#[llama_test(
    model_source = HuggingFaceShards(
        "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        "fbbb5b93fb787c21338159b0af3318bb3f4d9768",
        [
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf",
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00002-of-00003.gguf",
            "UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00003-of-00003.gguf",
        ]
    ),
    n_gpu_layers = 0,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_deepseek_v4_context_limits_positions_by_the_window_its_cache_masks_with(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let context = fixture.build_context()?;

    ensure!(context.limits().sliding_window == DEEPSEEK_V4_SLIDING_WINDOW);

    Ok(())
}
