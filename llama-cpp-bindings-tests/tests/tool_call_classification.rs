use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::get_weather_tools::get_weather_tools;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::visible_text_in::visible_text_in;
use llama_cpp_chat::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_model::sampled_token_section::SampledTokenSection;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const REASONING_BEFORE_THE_CALL: &str = "Let me check the weather.\n</think>\n\n";

const QWEN_XML_PARIS_WEATHER_CALL: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

const END_OF_GENERATION_TOKENS: u64 = 1;

fn raw_text_where(outcomes: &[IngestOutcome], belongs: impl Fn(&SampledToken) -> bool) -> String {
    outcomes
        .iter()
        .filter(|IngestOutcome { sampled_token, .. }| belongs(sampled_token))
        .map(|IngestOutcome { piece, .. }| piece.raw())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen35_reasoning_closed_by_a_tool_call_is_attributed_to_each_section(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(
        model,
        "What is the weather in Paris?",
        &get_weather_tools(),
        true,
    )?;
    let answer = format!("{REASONING_BEFORE_THE_CALL}{QWEN_XML_PARIS_WEATHER_CALL}");
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let outcomes = ingested_generation(&mut classifier, model, &answer)?;
    let parsed = match classifier
        .latest_reading()
        .context("the classifier must have read the answer")?
        .parse()
        .cloned()
        .map_err(Clone::clone)?
    {
        ChatMessageParseOutcome::Recognized(message) => message,
        ChatMessageParseOutcome::Unrecognized(raw_chat_message) => {
            bail!("the answer must be recognized: {raw_chat_message:?}")
        }
    };

    ensure!(
        raw_text_where(&outcomes, |token| matches!(
            token,
            SampledToken::Reasoning(_)
        )) == REASONING_BEFORE_THE_CALL
    );
    ensure!(
        raw_text_where(&outcomes, |token| matches!(
            token,
            SampledToken::ToolCall(_)
        )) == QWEN_XML_PARIS_WEATHER_CALL
    );
    ensure!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome.sampled_token, SampledToken::Reasoning(_)))
            .map(|outcome| outcome.piece.visible())
            .collect::<String>()
            == parsed.reasoning_content
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
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    const GENERATED_TEXT: &str = r#"{"name": "get_weather", "arguments": {"location": "Paris"}}"#;

    let session = user_turn_chat_session(model, "What is 2 + 2?", &[], false)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let outcomes = ingested_generation(&mut classifier, model, GENERATED_TEXT)?;

    ensure!(
        [
            visible_text_in(&outcomes, SampledTokenSection::Reasoning),
            visible_text_in(&outcomes, SampledTokenSection::Content),
            visible_text_in(&outcomes, SampledTokenSection::ToolCall),
            visible_text_in(&outcomes, SampledTokenSection::Pending),
        ] == [
            String::new(),
            GENERATED_TEXT.to_owned(),
            String::new(),
            String::new()
        ]
    );
    ensure!(
        classifier.usage().content
            == u64::try_from(outcomes.len())?
                .checked_add(END_OF_GENERATION_TOKENS)
                .context(
                    "`u64::try_from(outcomes.len())? + END_OF_GENERATION_TOKENS` must not overflow"
                )?
    );

    Ok(())
}
