use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use std::num::NonZeroU32;

use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_cpp_error::LlamaCppError;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::freshly_loaded_model::freshly_loaded_model;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_bindings_tests::non_causal_context::non_causal_context;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
use llama_cpp_bindings_tests::prompt_question::PROMPT_QUESTION;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_context_memory::context_memory::ContextMemory;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_mtmd::evaluated_mtmd_input_chunk::EvaluatedMtmdInputChunk;
use llama_cpp_mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_error::MtmdError;
use llama_cpp_mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const EVALUATED_IMAGE_DIMENSIONS: [ImageDimensions; 10] = [
    ImageDimensions {
        width: 224,
        height: 224,
    },
    ImageDimensions {
        width: 512,
        height: 512,
    },
    ImageDimensions {
        width: 100,
        height: 500,
    },
    ImageDimensions {
        width: 337,
        height: 421,
    },
    ImageDimensions {
        width: 2,
        height: 2,
    },
    ImageDimensions {
        width: 7,
        height: 13,
    },
    ImageDimensions {
        width: 3,
        height: 1000,
    },
    ImageDimensions {
        width: 1000,
        height: 3,
    },
    ImageDimensions {
        width: 1920,
        height: 1080,
    },
    ImageDimensions {
        width: 4096,
        height: 4096,
    },
];

fn eval_synthetic_bitmap(fixture: &LlamaFixture<'_>, width: u32, height: u32) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let image_data =
        vec![
            128u8;
            (width as usize)
                .checked_mul(height as usize)
                .context("`(width as usize) * (height as usize)` must not overflow")?
                .checked_mul(3)
                .context("`(width as usize) * (height as usize) * 3` must not overflow")?
        ];
    let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(
        n_positions
            .checked_add(256)
            .context("`n_positions + 256` must not overflow")?,
    )?;
    if fixture.context_params.n_ctx < required_n_ctx {
        bail!(
            "fixture n_ctx ({}) below required ({}) for {}x{} image",
            fixture.context_params.n_ctx,
            required_n_ctx,
            width,
            height,
        );
    }

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, false)? == n_positions,
        "evaluating a {width}x{height} image must advance past every chunk position"
    );
    Ok(())
}

fn tokenize_image_followed_by_long_text<'fixture>(
    fixture: &LlamaFixture<'fixture>,
) -> Result<MtmdInputChunks<'fixture>> {
    tokenize_prompt_with_synthetic_image(
        fixture,
        &format!("Describe: <__media__>{}", " llama".repeat(200)),
    )
}

fn tokenize_question_about_llamas<'context>(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
    mtmd_ctx: &'context MtmdContext,
) -> Result<MtmdInputChunks<'context>> {
    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    Ok(MtmdInputChunks::tokenize(
        mtmd_ctx,
        MtmdInputText {
            text: media_question_chat_session(model, PROMPT_QUESTION)?.prompt()?,
            add_special: true,
            parse_special: true,
        },
        &[&bitmap],
    )?)
}

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
fn eval_chunks_refuses_chunks_whose_positions_overflow_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let second_chunk = chunks.get(1).context("the image follows the first chunk")?;
    let mut llama_ctx = fixture.build_context()?;
    let max_position = llama_ctx.limits().max_position();
    let start_position = max_position
        .checked_sub(first_chunk.n_positions())
        .context("`max_position - first_chunk.n_positions()` must not overflow")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(first_chunk.position_after(&llama_ctx, start_position) == Ok(max_position));
    ensure!(
        chunks.eval_chunks(&mut llama_ctx, start_position, 0, n_batch, true)
            == Err(MtmdError::ChunkPositionsOverflow {
                start_position: max_position,
                n_tokens: second_chunk.n_tokens(),
                n_positions: second_chunk.n_positions(),
                max_position,
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
fn eval_chunks_reports_an_exception_llama_cpp_throws_while_evaluating_a_chunk(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS,
    );

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::EvalReported {
                message: injected_fault_message()?
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
fn eval_each_yields_nothing_after_the_first_chunk_that_fails(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__> briefly")?;

    ensure!(chunks.len() > 2, "a chunk must follow the one that fails");

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, n_batch, 0)?;
    let _fault = ArmedFault::after_passing(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS,
        1,
    );
    let mut next_positions = fitted_chunks
        .eval_each(&mut llama_ctx, 0, 0, n_batch, true)?
        .map(|evaluated_chunk| {
            evaluated_chunk.map(|EvaluatedMtmdInputChunk { next_position, .. }| next_position)
        });

    ensure!(
        next_positions.nth(1)
            == Some(Err(MtmdError::EvalReported {
                message: injected_fault_message()?
            }))
    );
    ensure!(next_positions.next().is_none());

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
fn fitting_no_chunks_leaves_nothing_to_evaluate(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let llama_ctx = fixture.build_context()?;
    let chunks = MtmdInputChunks::new(mtmd_ctx)?;
    let fitted_chunks = chunks.fit_in(&llama_ctx, i32::try_from(llama_ctx.n_batch())?, 0)?;

    ensure!(fitted_chunks.fitted_chunks().is_empty());

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
fn eval_chunks_rejects_a_sequence_the_context_does_not_have_instead_of_aborting(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let n_seq_max = llama_ctx.n_seq_max();
    let missing_sequence = i32::try_from(n_seq_max)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, missing_sequence, n_batch, true)
            == Err(MtmdError::SequenceIdOutOfRange {
                seq_id: missing_sequence,
                n_seq_max,
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
fn eval_chunks_rejects_a_context_created_for_another_model_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let another_model = freshly_loaded_model(fixture)?;
    let mut llama_ctx = LlamaContext::from_model(
        &another_model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, true)
            == Err(MtmdError::ContextModelIsNotTheProjectorModel)
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
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn chunk_eval_advances_past_every_position_of_the_chunk(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(first_chunk.eval(&mut llama_ctx, 0, 0, n_batch, true) == Ok(first_chunk.n_positions()));

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
fn chunk_eval_refuses_a_chunk_whose_positions_overflow_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>")?;
    let first_chunk = chunks.get(0).context("tokenization produced no chunks")?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(
        first_chunk.eval(&mut llama_ctx, i32::MAX, 0, n_batch, true)
            == Err(MtmdError::ChunkPositionsOverflow {
                start_position: i32::MAX,
                n_tokens: first_chunk.n_tokens(),
                n_positions: first_chunk.n_positions(),
                max_position: llama_ctx.limits().max_position(),
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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_advances_past_every_position_of_a_standard_image(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;

    let fixtures = fixtures_dir();
    let image_path = fixtures.join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;
    let input_text = MtmdInputText {
        text: "What is in this image? <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = MtmdInputChunks::tokenize(mtmd_ctx, input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(
        n_positions
            .checked_add(256)
            .context("`n_positions + 256` must not overflow")?,
    )?;
    ensure!(
        fixture.context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        fixture.context_params.n_ctx,
        required_n_ctx,
    );

    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    ensure!(chunks.eval_chunks(&mut llama_ctx, 0, 0, n_batch, false) == Ok(n_positions));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_evaluates_images_of_ordinary_and_extreme_dimensions(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    for ImageDimensions { width, height } in EVALUATED_IMAGE_DIMENSIONS {
        eval_synthetic_bitmap(fixture, width, height)
            .context(format!("dimension {width}x{height} should succeed"))?;
    }

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

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn fitting_reports_a_chunk_type_the_bindings_do_not_recognize(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .context("mmproj_file declared in attribute")?;
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let llama_ctx = fixture.build_context()?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNK_TYPE_UNKNOWN,
    );

    ensure!(
        chunks.fit_in(&llama_ctx, 256, 0).err()
            == Some(MtmdError::EvalUnknownChunkType(Box::new(
                MtmdError::UnknownInputChunkType(
                    llama_cpp_bindings_sys::bindings::MTMD_INPUT_CHUNK_TYPE_COUNT
                )
            )))
    );

    Ok(())
}

struct ImageDimensions {
    width: u32,
    height: u32,
}
