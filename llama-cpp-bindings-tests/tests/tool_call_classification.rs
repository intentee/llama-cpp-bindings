use anyhow::Result;

use llama_cpp_bindings::bare_json_tool_calls::BareJsonToolCalls;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::get_weather_tools_json::GET_WEATHER_TOOLS_JSON;
use llama_cpp_bindings_tests::ingested_generation::ingested_generation;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const REASONING_BEFORE_THE_CALL: &str = "Let me check the weather.\n</think>\n\n";
const QWEN_XML_PARIS_WEATHER_CALL: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

fn raw_text_where(outcomes: &[IngestOutcome], belongs: impl Fn(&SampledToken) -> bool) -> String {
    outcomes
        .iter()
        .filter(|outcome| belongs(&outcome.sampled_token))
        .map(|outcome| outcome.piece.raw())
        .collect()
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen35_reasoning_closed_by_a_tool_call_is_attributed_to_each_section(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        "What is the weather in Paris?",
        GET_WEATHER_TOOLS_JSON,
        true,
    )?;
    let answer = format!("{REASONING_BEFORE_THE_CALL}{QWEN_XML_PARIS_WEATHER_CALL}");
    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let outcomes = ingested_generation(&mut classifier, fixture.model, &answer);
    let parsed = recognized_answer(&session, &answer, ChatOutputCompleteness::Finished);

    assert_eq!(
        raw_text_where(&outcomes, |token| matches!(
            token,
            SampledToken::Reasoning(_)
        )),
        REASONING_BEFORE_THE_CALL
    );
    assert_eq!(
        raw_text_where(&outcomes, |token| matches!(
            token,
            SampledToken::ToolCall(_)
        )),
        QWEN_XML_PARIS_WEATHER_CALL
    );
    assert_eq!(
        outcomes
            .iter()
            .filter(|outcome| matches!(outcome.sampled_token, SampledToken::Reasoning(_)))
            .map(|outcome| outcome.piece.visible())
            .collect::<String>(),
        parsed.reasoning_content
    );

    Ok(())
}
