use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::error::eval_multimodal_chunks_error::EvalMultimodalChunksError;
use llama_cpp_bindings::error::sequence_id_out_of_range::SequenceIdOutOfRange;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::mtmd::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use llama_cpp_bindings::mtmd::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_context::MtmdContext;
use llama_cpp_bindings::mtmd::mtmd_eval_error::MtmdEvalError;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_bindings_tests::prime_memory_with::prime_memory_with;
use llama_cpp_bindings_tests::prompt_question::PROMPT_QUESTION;
use llama_cpp_bindings_tests::tokenize_prompt_with_synthetic_image::tokenize_prompt_with_synthetic_image;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message;
use llama_cpp_ffi_status::FfiContractError;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

fn eval_synthetic_bitmap(fixture: &LlamaFixture<'_>, width: u32, height: u32) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let image_data = vec![128u8; (width as usize) * (height as usize) * 3];
    let bitmap = MtmdBitmap::from_image_data(width, height, &image_data)?;
    let input_text = MtmdInputText {
        text: "Describe: <__media__>".to_string(),
        add_special: true,
        parse_special: true,
    };
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(n_positions + 256)?;
    if fixture.context_params.n_ctx < required_n_ctx {
        bail!(
            "fixture n_ctx ({}) below required ({}) for {}x{} image",
            fixture.context_params.n_ctx,
            required_n_ctx,
            width,
            height,
        );
    }

    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, n_batch, false)?;
    Ok(())
}

fn tokenize_question_about_llamas(
    fixture: &LlamaFixture<'_>,
    mtmd_ctx: &MtmdContext,
) -> Result<MtmdInputChunks> {
    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    Ok(mtmd_ctx.tokenize(
        MtmdInputText {
            text: media_question_chat_session(fixture.model, PROMPT_QUESTION)?.prompt()?,
            add_special: true,
            parse_special: true,
        },
        &[&bitmap],
    )?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 32,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_returns_batch_size_exceeds_context_limit_for_huge_batch(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let chunks = MtmdInputChunks::new()?;
    let huge_batch = i32::try_from(llama_ctx.n_batch() + 1)?;

    let result = chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, huge_batch, false);

    assert_eq!(
        result,
        Err(MtmdEvalError::BatchSizeExceedsContextLimit {
            requested: huge_batch,
            context_max: llama_ctx.n_batch(),
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_reports_an_exception_llama_cpp_throws_while_evaluating_a_chunk(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>");
    let mut llama_ctx = fixture.build_context()?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS);

    assert_eq!(
        chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, n_batch, true),
        Err(MtmdEvalError::Reported {
            message: injected_fault_message()
        })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>");
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert_eq!(
        chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, 0, true),
        Err(MtmdEvalError::NonPositiveBatchSize { requested: 0 })
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn fitting_no_chunks_leaves_nothing_to_evaluate(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let fitted_chunks =
        FittedMtmdInputChunks::fit(MtmdInputChunks::new()?, mtmd_ctx, llama_ctx.n_ubatch())?;

    assert!(fitted_chunks.is_empty());
    assert!(FittedMtmdInputChunk::at(&fitted_chunks, 0).is_none());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let fitted_chunks = FittedMtmdInputChunks::fit(
        tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>"),
        mtmd_ctx,
        llama_ctx.n_ubatch(),
    )?;
    let first_chunk =
        FittedMtmdInputChunk::at(&fitted_chunks, 0).context("tokenization produced no chunks")?;

    assert_eq!(
        first_chunk.eval_single(mtmd_ctx, &mut llama_ctx, 0, 0, 0, true),
        Err(MtmdEvalError::NonPositiveBatchSize { requested: 0 })
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_batch_larger_than_the_context_batch_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let mut llama_ctx = fixture.build_context()?;
    let fitted_chunks = FittedMtmdInputChunks::fit(
        tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>"),
        mtmd_ctx,
        llama_ctx.n_ubatch(),
    )?;
    let first_chunk =
        FittedMtmdInputChunk::at(&fitted_chunks, 0).context("tokenization produced no chunks")?;
    let context_max = llama_ctx.n_batch();
    let oversized_batch = i32::try_from(context_max + 1)?;

    assert_eq!(
        first_chunk.eval_single(mtmd_ctx, &mut llama_ctx, 0, 0, oversized_batch, true),
        Err(MtmdEvalError::BatchSizeExceedsContextLimit {
            requested: oversized_batch,
            context_max,
        })
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_rejects_a_sequence_the_context_does_not_have_instead_of_aborting(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let mut llama_ctx = fixture.build_context()?;
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>");
    let n_seq_max = llama_ctx.n_seq_max();
    let missing_sequence = i32::try_from(n_seq_max)?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;

    assert_eq!(
        chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, missing_sequence, n_batch, true),
        Err(MtmdEvalError::SequenceIdOutOfRange(SequenceIdOutOfRange {
            seq_id: missing_sequence,
            n_seq_max,
        }))
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_single_rejects_a_micro_batch_smaller_than_the_fitted_one_before_evaluating(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let fitted_micro_batch_tokens = llama_ctx.n_ubatch();
    let fitted_chunks = FittedMtmdInputChunks::fit(
        tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>"),
        mtmd_ctx,
        fitted_micro_batch_tokens,
    )?;
    let first_chunk =
        FittedMtmdInputChunk::at(&fitted_chunks, 0).context("tokenization produced no chunks")?;
    let smaller_micro_batch_tokens = fitted_micro_batch_tokens - 1;

    assert_eq!(
        first_chunk.eval_single(
            mtmd_ctx,
            &mut llama_ctx,
            0,
            0,
            i32::try_from(smaller_micro_batch_tokens)?,
            true
        ),
        Err(MtmdEvalError::EvaluatedMicroBatchSmallerThanFitted {
            evaluated_micro_batch_tokens: smaller_micro_batch_tokens,
            fitted_micro_batch_tokens,
        })
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn classifier_rejects_a_zero_batch_before_evaluating(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_prompt_with_synthetic_image(fixture, "Describe: <__media__>");
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    assert!(matches!(
        classifier.eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut llama_ctx,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 0,
                logits_last: true,
            },
        ),
        Err(EvalMultimodalChunksError::EvalFailed(
            MtmdEvalError::NonPositiveBatchSize { requested: 0 }
        ))
    ));
    assert_eq!(classifier.usage().prompt_tokens, 0);
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_succeeds_for_a_standard_image(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

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
    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let n_positions = chunks.total_positions();
    let required_n_ctx = u32::try_from(n_positions + 256)?;
    assert!(
        fixture.context_params.n_ctx >= required_n_ctx,
        "fixture n_ctx ({}) below required ({}); update the attribute literal",
        fixture.context_params.n_ctx,
        required_n_ctx,
    );

    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let result = chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, n_batch, false);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_succeeds_for_varied_image_dimensions(fixture: &LlamaFixture<'_>) {
    let test_dimensions: [(u32, u32); 4] = [(224, 224), (512, 512), (100, 500), (337, 421)];

    for (width, height) in test_dimensions {
        let result = eval_synthetic_bitmap(fixture, width, height);
        assert!(
            result.is_ok(),
            "dimension {width}x{height} should succeed: {result:?}"
        );
    }
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 8192,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "mmproj-F16.gguf"),
)]
fn eval_chunks_supports_extreme_dimensions(fixture: &LlamaFixture<'_>) -> Result<()> {
    let extreme_dimensions: [(u32, u32); 6] = [
        (2, 2),
        (7, 13),
        (3, 1000),
        (1000, 3),
        (1920, 1080),
        (4096, 4096),
    ];

    for (width, height) in extreme_dimensions {
        eval_synthetic_bitmap(fixture, width, height)
            .context(format!("dimension {width}x{height} should succeed"))?;
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let image_tokens = ChunkTokenBreakdown::from_chunks(&chunks).image;
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let micro_batch_tokens = llama_ctx.n_ubatch();

    assert_eq!(
        chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, 0, 0, n_batch, true),
        Err(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
            NonCausalChunkMicroBatchMismatch {
                chunk_tokens: usize::try_from(image_tokens)?,
                micro_batch_tokens,
            }
        ))
    );
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let image_chunk = (0..chunks.len())
        .filter_map(|chunk_index| chunks.get(chunk_index))
        .find(|chunk| chunk.chunk_type() == Ok(MtmdInputChunkType::Image))
        .context("the prompt must contain an image chunk")?;
    let llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    assert_eq!(
        image_chunk.fit_to_micro_batch(mtmd_ctx, llama_ctx.n_ubatch()),
        Err(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
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
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    let n_batch = i32::try_from(llama_ctx.n_batch())?;
    let evaluation = classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut llama_ctx,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch,
            logits_last: true,
        },
    );

    assert!(matches!(
        evaluation,
        Err(EvalMultimodalChunksError::EvalFailed(
            MtmdEvalError::NonCausalChunkExceedsMicroBatch(NonCausalChunkMicroBatchMismatch {
                micro_batch_tokens,
                ..
            })
        )) if micro_batch_tokens == llama_ctx.n_ubatch()
    ));
    assert_eq!(classifier.usage().prompt_tokens, 0);
    assert_eq!(llama_ctx.memory_seq_pos_max(0)?, -1);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let breakdown = ChunkTokenBreakdown::from_chunks(&chunks);
    let mut llama_ctx = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    prime_memory_with(fixture, &mut llama_ctx, &"hello ".repeat(230))?;

    let primed_positions = llama_ctx.memory_seq_pos_max(0)? + 1;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    assert!(
        u64::from(llama_ctx.n_ctx()) - u64::try_from(primed_positions)? < breakdown.image,
        "the primed context must have less room left than the image needs"
    );
    assert!(matches!(
        classifier.eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut llama_ctx,
            EvalMultimodalChunksParams {
                start_position: primed_positions,
                seq_id: 0,
                n_batch: 256,
                logits_last: true,
            },
        ),
        Err(EvalMultimodalChunksError::EvalFailed(
            MtmdEvalError::EvalFailed { code: 1 }
        ))
    ));
    assert_eq!(classifier.usage().input_image_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = fixture.build_context()?;

    prime_memory_with(fixture, &mut llama_ctx, &"hello ".repeat(230))?;

    let primed_positions = llama_ctx.memory_seq_pos_max(0)? + 1;

    assert_eq!(
        chunks.eval_chunks(mtmd_ctx, &mut llama_ctx, primed_positions, 0, 256, true),
        Err(MtmdEvalError::EvalFailed { code: 1 })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn fitting_reports_a_chunk_llama_cpp_does_not_return(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_GET_RETURNS_NULL,
    );

    assert_eq!(
        chunks.fit_to_micro_batch(mtmd_ctx, 256),
        Err(MtmdEvalError::FfiContract(FfiContractError {
            operation: "mtmd_input_chunks_get",
            detail: "returned a null chunk within the chunk count",
        }))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNK_TYPE_UNKNOWN);

    assert_eq!(
        chunks.fit_to_micro_batch(mtmd_ctx, 256),
        Err(MtmdEvalError::UnknownChunkType(MtmdInputChunkTypeError(
            llama_cpp_bindings_sys::MTMD_INPUT_CHUNK_TYPE_COUNT
        )))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 256,
    n_ubatch = 256,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn classifier_reports_a_fitted_chunk_llama_cpp_no_longer_returns(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");
    let chunks = tokenize_question_about_llamas(fixture, mtmd_ctx)?;
    let mut llama_ctx = fixture.build_context()?;
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);
    let _fault = ArmedFault::after_passing(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_GET_RETURNS_NULL,
        chunks.len(),
    );

    assert_eq!(
        classifier.eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut llama_ctx,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 256,
                logits_last: true,
            },
        ),
        Err(EvalMultimodalChunksError::ChunkOutOfBounds(0))
    );

    Ok(())
}
