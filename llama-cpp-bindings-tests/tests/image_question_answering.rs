use anyhow::Context;
use anyhow::Result;

use llama_cpp_bindings::bare_json_tool_calls::BareJsonToolCalls;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_capability::MtmdCapability;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf"),
)]
fn a_question_about_an_image_is_answered_with_its_image_tokens_counted_as_input(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    assert!(
        mtmd_ctx
            .capabilities(model)
            .contains(MtmdCapability::Vision),
        "model should support vision input"
    );

    let image_path = fixtures_dir().join("llamas.jpg");
    let image_path_str = image_path
        .to_str()
        .with_context(|| "image path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, image_path_str)
        .with_context(|| "failed to load image from file")?;

    let session = media_question_chat_session(model, "What animals do you see in this image?")?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let expected = ChunkTokenBreakdown::from_chunks(&chunks);

    assert!(
        expected.image > 0,
        "vision input must produce at least one image chunk"
    );

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut ctx,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 512,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate chunks")?;

    {
        let usage = classifier.usage();
        assert_eq!(usage.prompt_tokens, expected.text);
        assert_eq!(usage.input_image_tokens, expected.image);
        assert_eq!(usage.input_audio_tokens, expected.audio);
    }

    let mut sampler = LlamaSampler::greedy()?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&ctx));
    let totals = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut ctx,
        batch: &mut batch,
        initial_position: n_past,
        max_generated_tokens: 512,
    }
    .run();

    assert!(
        !totals.generated.text().is_empty(),
        "model should generate at least one token from image input"
    );

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, expected.text);
    assert_eq!(usage.input_image_tokens, expected.image);
    assert_eq!(usage.input_audio_tokens, expected.audio);
    assert_eq!(usage.content_tokens, totals.observed_content);
    assert_eq!(usage.reasoning_tokens, totals.observed_reasoning);
    assert_eq!(
        usage.completion_tokens(),
        totals.observed_content + totals.observed_reasoning
    );

    Ok(())
}
