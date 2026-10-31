use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::mtmd::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use llama_cpp_bindings::mtmd::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::prompt_question::PROMPT_QUESTION;
use llama_cpp_bindings_types::TokenUsage;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

fn build_multimodal_chunks_and_eval_into_usage(
    fixture: &LlamaFixture<'_>,
) -> Result<(TokenUsage, ChunkTokenBreakdown)> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker()?;
    let prompt = format!("{marker}{PROMPT_QUESTION}");

    let input_text = MtmdInputText {
        text: prompt,
        add_special: false,
        parse_special: true,
    };

    let chunks = mtmd_ctx.tokenize(input_text, &[&bitmap])?;
    let expected = ChunkTokenBreakdown::from_chunks(&chunks);

    let context_params = (*fixture.context_params).into_llama_context_params();
    let mut context = LlamaContext::from_model(model, fixture.backend, context_params)?;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    classifier.eval_multimodal_chunks(
        &chunks,
        mtmd_ctx,
        &mut context,
        EvalMultimodalChunksParams {
            start_position: 0,
            seq_id: 0,
            n_batch: 512,
            logits_last: true,
        },
    )?;

    Ok((classifier.into_usage(), expected))
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn prompt_tokens_match_text_chunk_total(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.prompt_tokens != expected.text {
        bail!(
            "prompt_tokens must equal sum of text-chunk n_tokens; expected {}, got {}",
            expected.text,
            usage.prompt_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn input_image_tokens_match_image_chunk_total(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.input_image_tokens != expected.image {
        bail!(
            "input_image_tokens must equal sum of image-chunk n_tokens; expected {}, got {}",
            expected.image,
            usage.input_image_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn input_audio_tokens_are_zero_for_image_only_input(fixture: &LlamaFixture<'_>) -> Result<()> {
    let (usage, expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if expected.audio != 0 {
        bail!(
            "fixture invariant: image-only multimodal input should produce zero audio chunk tokens, got {}",
            expected.audio
        );
    }
    if usage.input_audio_tokens != 0 {
        bail!(
            "input_audio_tokens must be zero when no audio chunks are evaluated; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn completion_tokens_are_zero_after_eval_before_generation(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let (usage, _expected) = build_multimodal_chunks_and_eval_into_usage(fixture)?;

    if usage.completion_tokens() != 0 {
        bail!(
            "completion_tokens must be zero immediately after eval (no generation has occurred); got {}",
            usage.completion_tokens()
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn text_chunk_records_prompt_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let input_text = MtmdInputText {
        text: "hello world".to_owned(),
        add_special: false,
        parse_special: false,
    };
    let llama_ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let fitted_chunks = FittedMtmdInputChunks::fit(
        mtmd_ctx.tokenize(input_text, &[])?,
        mtmd_ctx,
        llama_ctx.n_ubatch(),
    )?;

    let text_chunk = (0..fitted_chunks.len())
        .filter_map(|index| FittedMtmdInputChunk::at(&fitted_chunks, index))
        .find(|chunk| chunk.chunk_type() == MtmdInputChunkType::Text)
        .context("text-only tokenization should produce at least one text chunk")?;

    let n_tokens = u64::try_from(text_chunk.chunk().n_tokens())?;

    let mut classifier = SampledTokenClassifier::undeterminable(model);

    classifier.ingest_fitted_chunk(&text_chunk);

    let usage = classifier.usage();
    if usage.prompt_tokens != n_tokens {
        bail!(
            "text chunk must record n_tokens as prompt_tokens; expected {n_tokens}, got {}",
            usage.prompt_tokens
        );
    }
    if usage.input_image_tokens != 0 {
        bail!(
            "text chunk must not bump input_image_tokens; got {}",
            usage.input_image_tokens
        );
    }
    if usage.input_audio_tokens != 0 {
        bail!(
            "text chunk must not bump input_audio_tokens; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn image_chunk_records_input_image_tokens_only(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .context("image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)?;

    let marker = mtmd_default_marker()?;
    let input_text = MtmdInputText {
        text: marker.to_owned(),
        add_special: false,
        parse_special: true,
    };
    let llama_ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let fitted_chunks = FittedMtmdInputChunks::fit(
        mtmd_ctx.tokenize(input_text, &[&bitmap])?,
        mtmd_ctx,
        llama_ctx.n_ubatch(),
    )?;

    let image_chunk = (0..fitted_chunks.len())
        .filter_map(|index| FittedMtmdInputChunk::at(&fitted_chunks, index))
        .find(|chunk| chunk.chunk_type() == MtmdInputChunkType::Image)
        .context("multimodal tokenization should produce an image chunk")?;

    let n_tokens = u64::try_from(image_chunk.chunk().n_tokens())?;
    if n_tokens == 0 {
        bail!("image chunk should report at least one token");
    }

    let mut classifier = SampledTokenClassifier::undeterminable(model);

    classifier.ingest_fitted_chunk(&image_chunk);

    let usage = classifier.usage();
    if usage.input_image_tokens != n_tokens {
        bail!(
            "image chunk must record n_tokens as input_image_tokens; expected {n_tokens}, got {}",
            usage.input_image_tokens
        );
    }
    if usage.prompt_tokens != 0 {
        bail!(
            "image chunk must not bump prompt_tokens; got {}",
            usage.prompt_tokens
        );
    }
    if usage.input_audio_tokens != 0 {
        bail!(
            "image chunk must not bump input_audio_tokens; got {}",
            usage.input_audio_tokens
        );
    }

    Ok(())
}
