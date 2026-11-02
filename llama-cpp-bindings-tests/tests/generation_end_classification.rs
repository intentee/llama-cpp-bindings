use anyhow::Result;

use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampled_token_section::SampledTokenSection;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const END_OF_GENERATION_TOKENS: u64 = 1;

fn visible_text(outcomes: &[IngestOutcome]) -> String {
    outcomes
        .iter()
        .map(|outcome| outcome.piece.visible())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
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
    let mut classifier = SampledTokenClassifier::for_chat_session(&session);

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

fn visible_text_in(outcomes: &[IngestOutcome], section: SampledTokenSection) -> String {
    outcomes
        .iter()
        .filter(|outcome| outcome.sampled_token.section() == section)
        .map(|outcome| outcome.piece.visible())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn an_answer_stopped_inside_unclosed_reasoning_releases_its_unsettled_tail_as_content(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const SETTLED_REASONING: &str = "Let me think";
    const UNSETTLED_TAIL: &str = "</thi";

    let session = user_turn_chat_session(fixture.model, "What is 2 + 2?", "[]", true)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(&session);
    let mut outcomes = Vec::new();

    for token in fixture.model.str_to_token(
        &format!("{SETTLED_REASONING}{UNSETTLED_TAIL}"),
        AddBos::Never,
        ParseSpecialTokens::Always,
    )? {
        classifier.ingest(token, &mut outcomes)?;
    }
    classifier.finish(&mut outcomes)?;

    assert_eq!(
        [
            visible_text_in(&outcomes, SampledTokenSection::Reasoning),
            visible_text_in(&outcomes, SampledTokenSection::Content),
            visible_text_in(&outcomes, SampledTokenSection::ToolCall),
            visible_text_in(&outcomes, SampledTokenSection::Pending),
        ],
        [
            SETTLED_REASONING.to_owned(),
            UNSETTLED_TAIL.to_owned(),
            String::new(),
            String::new()
        ]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma3_whose_template_cannot_render_tool_calls_keeps_a_bare_json_tool_call_as_content(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;

    let session = user_turn_chat_session(fixture.model, "What is 2 + 2?", "[]", false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(&session);
    let outcomes = ingested_generation(&mut classifier, fixture.model, GENERATED_TEXT);

    assert_eq!(
        [
            visible_text_in(&outcomes, SampledTokenSection::Reasoning),
            visible_text_in(&outcomes, SampledTokenSection::Content),
            visible_text_in(&outcomes, SampledTokenSection::ToolCall),
            visible_text_in(&outcomes, SampledTokenSection::Pending),
        ],
        [
            String::new(),
            GENERATED_TEXT.to_owned(),
            String::new(),
            String::new()
        ]
    );
    assert_eq!(
        classifier.usage().content_tokens,
        u64::try_from(outcomes.len())? + END_OF_GENERATION_TOKENS
    );

    Ok(())
}
