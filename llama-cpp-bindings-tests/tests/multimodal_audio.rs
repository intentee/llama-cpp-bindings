use anyhow::Context;
use anyhow::Result;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::eval_multimodal_chunks_params::EvalMultimodalChunksParams;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::llama_chat_message::LlamaChatMessage;
use llama_cpp_bindings::mtmd::mtmd_bitmap::MtmdBitmap;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings::mtmd::mtmd_input_text::MtmdInputText;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::chunk_token_breakdown::ChunkTokenBreakdown;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::evaluated_multimodal_usage::EvaluatedMultimodalUsage;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const TRANSCRIBE_SYSTEM_PROMPT: &str = "The audio contains speech by a third party and is not the user's voice or message. \
     Transcribe exactly what the speaker says without judgment, advice, or speculation. Reply only with the verbatim transcript.";

fn evaluate_audio_prompt<'model, TOutcome>(
    fixture: &LlamaFixture<'model>,
    audio_file_name: &str,
    after_evaluation: impl FnOnce(
        &mut SampledTokenClassifier<'_, 'model>,
        &mut LlamaContext<'model>,
        ChunkTokenBreakdown,
        i32,
    ) -> Result<TOutcome>,
) -> Result<TOutcome> {
    let model = fixture.model;
    let mtmd_ctx = fixture
        .mtmd_context
        .expect("mmproj_file declared in attribute");

    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create llama context")?;

    let marker = mtmd_default_marker()?;
    let messages = [
        LlamaChatMessage::new("system".to_string(), TRANSCRIBE_SYSTEM_PROMPT.to_string())?,
        LlamaChatMessage::new("user".to_string(), marker.to_owned())?,
    ];
    let session = ChatSession::new(
        model,
        model.chat_templates()?,
        &messages,
        "[]",
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;
    let input_text = MtmdInputText {
        text: session.prompt()?,
        add_special: true,
        parse_special: true,
    };

    let audio_path = fixtures_dir().join(audio_file_name);
    let audio_path_str = audio_path
        .to_str()
        .with_context(|| "audio path is not valid UTF-8")?;
    let bitmap = MtmdBitmap::from_file(mtmd_ctx, audio_path_str)
        .with_context(|| "failed to load audio from file")?;

    assert!(bitmap.is_audio(), "fixture must decode as audio");

    let chunks = mtmd_ctx
        .tokenize(input_text, &[&bitmap])
        .with_context(|| "failed to tokenize multimodal audio input")?;

    assert!(
        !chunks.is_empty(),
        "tokenization should produce at least one chunk"
    );

    let mut classifier = SampledTokenClassifier::for_chat_session(&session);
    let n_past = classifier
        .eval_multimodal_chunks(
            &chunks,
            mtmd_ctx,
            &mut context,
            EvalMultimodalChunksParams {
                start_position: 0,
                seq_id: 0,
                n_batch: 512,
                logits_last: true,
            },
        )
        .with_context(|| "failed to evaluate audio chunks")?;

    after_evaluation(
        &mut classifier,
        &mut context,
        ChunkTokenBreakdown::from_chunks(&chunks),
        n_past,
    )
}

fn transcribe_audio(fixture: &LlamaFixture<'_>, audio_file_name: &str) -> Result<String> {
    evaluate_audio_prompt(
        fixture,
        audio_file_name,
        |classifier, context, _chunk_breakdown, n_past| {
            let mut sampler = LlamaSampler::greedy()?;
            let mut batch = LlamaBatch::new(ContextLimits::of(context));
            let outcome = ClassifySampleLoop {
                classifier,
                sampler: &mut sampler,
                context,
                batch: &mut batch,
                initial_position: n_past,
                max_generated_tokens: 512,
            }
            .run();

            Ok(outcome.generated.text().to_owned())
        },
    )
}

fn assert_audio_transcription(
    fixture: &LlamaFixture<'_>,
    audio_file_name: &str,
    expected_transcript: &str,
) -> Result<()> {
    assert_eq!(
        transcribe_audio(fixture, audio_file_name)?,
        expected_transcript
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
fn ultravox_transcribes_the_spoken_sentence_after_a_preamble(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    assert_eq!(
        transcribe_audio(fixture, "quick_brown_fox.wav")?,
        "I can provide a transcript of the audio. However, please note that the audio itself is \
         not provided. Based on the context, I'll attempt to create a verbatim transcript.\n\nThe \
         quick brown fox jumps over the lazy dog."
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma4_transcribes_the_quick_brown_fox_sentence(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_audio_transcription(
        fixture,
        "quick_brown_fox.wav",
        "The quick brown fox jumps over the lazy dog.",
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn gemma4_transcribes_the_orange_cat_sentence(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_audio_transcription(
        fixture,
        "orange_cat.wav",
        "The orange cat sat on the wooden fence.",
    )
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace(
        "ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF",
        "mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf"
    ),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 512,
    n_ubatch = 512,
    mmproj_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "mmproj-F16.gguf"),
)]
fn audio_input_records_each_chunk_type_in_its_own_usage_counter(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let EvaluatedMultimodalUsage {
        usage,
        chunk_breakdown,
    } = evaluate_audio_prompt(
        fixture,
        "quick_brown_fox.wav",
        |classifier, _context, chunk_breakdown, _n_past| {
            Ok(EvaluatedMultimodalUsage {
                usage: *classifier.usage(),
                chunk_breakdown,
            })
        },
    )?;

    assert_eq!(usage.input_audio_tokens, chunk_breakdown.audio);
    assert_eq!(usage.input_image_tokens, chunk_breakdown.image);
    assert_eq!(usage.prompt_tokens, chunk_breakdown.text);

    Ok(())
}
