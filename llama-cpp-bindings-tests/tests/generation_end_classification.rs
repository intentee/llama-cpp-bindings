use anyhow::Result;

use llama_cpp_bindings::bare_json_tool_calls::BareJsonToolCalls;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const END_OF_GENERATION_TOKENS: u64 = 1;

fn visible_text(outcomes: &[IngestOutcome]) -> String {
    outcomes
        .iter()
        .map(|outcome| outcome.piece.visible())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_classifier_ends_generation_without_emitting_the_end_of_generation_token(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = "The answer is four.";

    let session = user_turn_chat_session(fixture.model, "What is 2 + 2?", "[]", false)?;
    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);

    assert_eq!(
        visible_text(&ingested_generation(
            &mut classifier,
            fixture.model,
            GENERATED_TEXT
        )),
        GENERATED_TEXT
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma3_releases_a_held_bare_json_prefix_as_content_when_generation_ends(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = r#"{"answer": 4"#;

    let session = user_turn_chat_session(fixture.model, "What is 2 + 2?", "[]", false)?;
    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let outcomes = ingested_generation(&mut classifier, fixture.model, GENERATED_TEXT);

    assert!(!session.supports_tool_calls());
    assert_eq!(visible_text(&outcomes), GENERATED_TEXT);
    assert!(
        outcomes
            .iter()
            .all(|outcome| matches!(outcome.sampled_token, SampledToken::Content(_))),
        "{outcomes:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma3_attributes_a_bare_json_tool_call_object_to_the_tool_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;

    let session = user_turn_chat_session(fixture.model, "What is 2 + 2?", "[]", false)?;
    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let outcomes = ingested_generation(&mut classifier, fixture.model, GENERATED_TEXT);

    assert!(
        outcomes
            .iter()
            .all(|outcome| matches!(outcome.sampled_token, SampledToken::ToolCall(_))),
        "{outcomes:?}"
    );
    assert_eq!(
        classifier.usage().tool_call_tokens,
        u64::try_from(outcomes.len())? + END_OF_GENERATION_TOKENS
    );

    Ok(())
}
