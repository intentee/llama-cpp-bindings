use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::evaluate_audio_prompt::evaluate_audio_prompt;
use llama_cpp_chat::chat_assistant_prefill::ChatAssistantPrefill;
use llama_cpp_chat::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn transcribe_audio(
    fixture: &LlamaFixture<'_>,
    audio_file_name: &str,
    assistant_turn: ChatAssistantTurn,
) -> Result<String> {
    evaluate_audio_prompt(
        fixture,
        audio_file_name,
        assistant_turn,
        |classifier, context, _chunk_breakdown, n_past| {
            let mut sampler = LlamaSampler::greedy()?;
            let mut batch = LlamaBatch::new(context.limits());
            let outcome = ClassifySampleLoop {
                classifier,
                sampler: &mut sampler,
                context,
                batch: &mut batch,
                initial_position: n_past,
                max_generated_tokens: 512,
            }
            .run()?;

            Ok(outcome.generated.text().to_owned())
        },
    )
}

fn assert_audio_transcription(
    fixture: &LlamaFixture<'_>,
    audio_file_name: &str,
    expected_transcript: &str,
) -> Result<()> {
    ensure!(
        transcribe_audio(fixture, audio_file_name, ChatAssistantTurn::New)? == expected_transcript
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
fn ultravox_transcribes_the_spoken_sentence_into_an_opened_quote(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(
        transcribe_audio(
            fixture,
            "quick_brown_fox.wav",
            ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
                reasoning_content: String::new(),
                content: "\"".to_owned(),
            }),
        )? == "Quick brown fox jumps over the lazy dog.\""
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
