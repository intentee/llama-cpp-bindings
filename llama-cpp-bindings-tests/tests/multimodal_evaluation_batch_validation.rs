use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_refuses_a_batch_larger_than_the_context_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let mut llama_ctx = fixture.build_context()?;

    let chunks = MtmdInputChunks::new(mtmd_ctx)?;
    let huge_batch = i32::try_from(
        llama_ctx
            .n_batch()
            .checked_add(1)
            .context("`llama_ctx.n_batch() + 1` must not overflow")?,
    )?;

    let result = chunks.eval_chunks(&mut llama_ctx, 0, 0, huge_batch, false);

    ensure!(
        result
            == Err(MtmdError::BatchSizeExceedsContextLimit {
                requested: huge_batch,
                context_max: llama_ctx.n_batch(),
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let mut llama_ctx = fixture.build_context()?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, 0, true)
            == Err(MtmdError::NonPositiveBatchSize { requested: 0 })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(llama_ctx.n_batch())?, 0)?;
    let first_chunk = fitted_chunks
        .fitted_chunks()
        .first()
        .context("tokenization produced no chunks")?;

    ensure!(
        first_chunk.eval_single(&mut llama_ctx, 0, 0, 0, true)
            == Err(MtmdError::NonPositiveBatchSize { requested: 0 })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_batch_larger_than_the_context_batch_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let context_max = llama_ctx.n_batch();
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(context_max)?, 0)?;
    let first_chunk = fitted_chunks
        .fitted_chunks()
        .first()
        .context("tokenization produced no chunks")?;
    let oversized_batch = i32::try_from(
        context_max
            .checked_add(1)
            .context("`context_max + 1` must not overflow")?,
    )?;

    ensure!(
        first_chunk.eval_single(&mut llama_ctx, 0, 0, oversized_batch, true)
            == Err(MtmdError::BatchSizeExceedsContextLimit {
                requested: oversized_batch,
                context_max,
            })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_eval_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;

    ensure!(
        first_chunk.eval(&mut llama_ctx, 0, 0, 0, true)
            == Err(MtmdError::NonPositiveBatchSize { requested: 0 })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn classifier_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let mut llama_ctx = fixture.build_context()?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    ensure!(matches!(
        classifier.eval_multimodal_chunks(
            &chunks,
            &mut llama_ctx,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 0,
                logits_last: true,
            },
        ),
        Err(LlamaCppError::MultimodalChunksEval(
            MtmdError::NonPositiveBatchSize { requested: 0 }
        ))
    ));
    ensure!(classifier.usage().prompt == 0);
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}
