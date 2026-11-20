use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
use llama_cpp_bindings_tests::tokenize_question_about_llamas::tokenize_question_about_llamas;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn classifier_reports_an_image_the_context_has_no_room_left_for(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let breakdown = ChunkTokenBreakdown::from_chunks(&chunks)?;
    let mut llama_ctx = fixture.build_context()?;

    prime_memory_with(fixture, &mut llama_ctx, &"hello ".repeat(230))?;

    let primed_positions = llama_ctx
        .memory_seq_pos_max(0)?
        .checked_add(1)
        .context("`llama_ctx.memory_seq_pos_max(0)? + 1` must not overflow")?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    ensure!(
        u64::from(llama_ctx.n_ctx()).checked_sub(u64::try_from(primed_positions)?).context("`u64::from(llama_ctx.n_ctx()) - u64::try_from(primed_positions)?` must not overflow")? < breakdown.image,
        "the primed context must have less room left than the image needs"
    );
    ensure!(matches!(
        classifier.eval_multimodal_chunks(
            &chunks,
            &mut llama_ctx,
            EvalMultimodalChunksParams {
                start_position: primed_positions,
                seq_id: 0,
                n_batch: 256,
                logits_last: true,
            },
        ),
        Err(LlamaCppError::MultimodalChunksEval(MtmdError::EvalFailed {
            code: 1
        }))
    ));
    ensure!(classifier.usage().input_image == 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_reports_an_image_the_context_has_no_room_left_for(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = fixture.build_context()?;

    prime_memory_with(fixture, &mut llama_ctx, &"hello ".repeat(230))?;

    let primed_positions = llama_ctx
        .memory_seq_pos_max(0)?
        .checked_add(1)
        .context("`llama_ctx.memory_seq_pos_max(0)? + 1` must not overflow")?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, primed_positions, 0, 256, true)
            == Err(MtmdError::EvalFailed { code: 1 })
    );

    Ok(())
}
