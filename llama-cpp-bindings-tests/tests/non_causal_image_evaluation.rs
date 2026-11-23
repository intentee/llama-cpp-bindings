use std::num::NonZeroU32;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::non_causal_context::non_causal_context;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_bindings_tests::tokenize_question_about_llamas::tokenize_question_about_llamas;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn tokenize_image_followed_by_long_text<'fixture>(
    fixture: &LlamaFixture<'fixture>,
) -> Result<MtmdInputChunks<'fixture>> {
    tokenize_prompt_with_synthetic_image(
        fixture,
        &format!("Describe: <__media__>{}", " llama".repeat(200)),
    )
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
fn eval_chunks_refuses_a_chunk_whose_decode_piece_exceeds_the_micro_batch_of_a_non_causal_context_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_image_followed_by_long_text(fixture)?;
    let last_chunk = chunks
        .chunks()
        .last()
        .copied()
        .context("tokenization produced no chunks")?;
    let mut llama_ctx = non_causal_context(fixture)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::ChunkPieceExceedsSingleMicroBatch {
                chunk_tokens: last_chunk.n_tokens(),
                piece_tokens: llama_ctx.n_batch(),
                micro_batch: llama_ctx.n_ubatch(),
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
fn eval_chunks_advances_past_chunks_whose_decode_pieces_fit_the_micro_batch_of_a_non_causal_context(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let mut llama_ctx = non_causal_context(fixture)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        llama_ctx.n_batch() > llama_ctx.n_ubatch(),
        "a decode of the whole batch must not fit the micro batch"
    );
    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true) == Ok(chunks.total_positions())
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
fn eval_single_refuses_a_batch_whose_decode_piece_exceeds_the_micro_batch_of_a_non_causal_context_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_image_followed_by_long_text(fixture)?;
    let mut llama_ctx = non_causal_context(fixture)?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(llama_ctx.n_ubatch())?, 0)?;
    let last_fitted_chunk = fitted_chunks
        .fitted_chunks()
        .last()
        .context("tokenization produced no chunks")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        last_fitted_chunk.eval_single(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::ChunkPieceExceedsSingleMicroBatch {
                chunk_tokens: last_fitted_chunk.chunk().n_tokens(),
                piece_tokens: llama_ctx.n_batch(),
                micro_batch: llama_ctx.n_ubatch(),
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
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_micro_batch_smaller_than_the_fitted_one_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let fitted_micro_batch_tokens = llama_ctx.n_ubatch();
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(llama_ctx.n_batch())?, 0)?;
    let first_chunk = fitted_chunks
        .fitted_chunks()
        .first()
        .context("tokenization produced no chunks")?;
    let smaller_micro_batch_tokens = fitted_micro_batch_tokens
        .checked_sub(1)
        .context("`fitted_micro_batch_tokens - 1` must not overflow")?;

    ensure!(
        first_chunk.eval_single(
            &mut llama_ctx,
            0,
            0,
            i32::try_from(smaller_micro_batch_tokens)?,
            true
        ) == Err(MtmdError::EvaluatedMicroBatchSmallerThanFitted {
            evaluated_micro_batch_tokens: smaller_micro_batch_tokens,
            fitted_micro_batch_tokens,
        })
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma3_decoding_follows_the_causal_attention_the_mtmd_helper_leaves_on_the_context(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = non_causal_context(fixture)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let next_position = chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, false)?;
    let tokens = fixture.model.str_to_token(
        &"llama ".repeat(600),
        AddBos::Never,
        ParseSpecialTokens::Never,
    )?;
    let last_index = tokens
        .len()
        .checked_sub(1)
        .context("`tokens.len() - 1` must not overflow")?;
    let mut batch = LlamaBatch::new(llama_ctx.limits());

    for (index, token) in tokens.iter().enumerate() {
        batch.add(
            SampledToken::Content(*token),
            next_position
                .checked_add(i32::try_from(index)?)
                .context("`next_position + i32::try_from(index)?` must not overflow")?,
            &[0],
            index == last_index,
        )?;
    }

    ensure!(batch.n_tokens().unsigned_abs() > llama_ctx.n_ubatch());
    ensure!(llama_ctx.decode(&batch) == Ok(()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 128,
    mmproj_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma3_eval_chunks_rejects_a_non_causal_image_larger_than_the_micro_batch_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let image_tokens = ChunkTokenBreakdown::from_chunks(&chunks)?.image;
    let mut llama_ctx = fixture.build_context()?;

    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let micro_batch_tokens = llama_ctx.n_ubatch();

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: usize::try_from(image_tokens)?,
                    micro_batch_tokens,
                }
            ))
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 128,
    mmproj_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma3_chunk_eval_rejects_a_non_causal_image_larger_than_the_micro_batch_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let image_chunk = chunks
        .chunks()
        .into_iter()
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Image))
        .context("the prompt must contain an image chunk")?;
    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let micro_batch_tokens = llama_ctx.n_ubatch();

    ensure!(
        image_chunk.eval(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: image_chunk.n_tokens(),
                    micro_batch_tokens,
                }
            ))
    );
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 128,
    mmproj_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma3_fitting_rejects_a_non_causal_image_larger_than_the_micro_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let image_chunk = chunks
        .chunks()
        .into_iter()
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Image))
        .context("the prompt must contain an image chunk")?;
    let llama_ctx = fixture.build_context()?;
    let batch_tokens =
        NonZeroU32::new(llama_ctx.n_batch()).context("the context batch must be positive")?;

    ensure!(
        image_chunk.fit_to_micro_batch(&llama_ctx, batch_tokens)
            == Err(MtmdError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: image_chunk.n_tokens(),
                    micro_batch_tokens: llama_ctx.n_ubatch(),
                }
            ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 128,
    mmproj_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma3_classifier_leaves_the_prompt_untouched_when_a_non_causal_image_exceeds_the_micro_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = fixture.build_context()?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let evaluation = classifier.eval_multimodal_chunks(
        &chunks,
        &mut llama_ctx,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch,
            logits_last: true,
        },
    );

    ensure!(matches!(
        evaluation,
        Err(LlamaCppError::MultimodalChunksEval(
            MtmdError::NonCausalChunkExceedsMicroBatch(NonCausalChunkMicroBatchMismatch {
                micro_batch_tokens,
                ..
            })
        )) if micro_batch_tokens == llama_ctx.n_ubatch()
    ));
    ensure!(classifier.usage().prompt == 0);
    ensure!(llama_ctx.memory_seq_pos_max(0)? == -1);

    Ok(())
}
