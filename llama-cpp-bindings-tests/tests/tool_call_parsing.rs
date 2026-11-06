use anyhow::Result;
use serde_json::json;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::raw_chat_message::RawChatMessage;
use llama_cpp_bindings_tests::get_weather_tools::get_weather_tools;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_types::chat_tool::ChatTool;
use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;
use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const WEATHER_QUESTION: &str = "What is the weather in Paris?";
const MALFORMED_QWEN_XML_TOOL_CALL: &str = "<tool_call>garbage</tool_call>";
const PLAIN_CONTENT_REFUSAL: &str = "Sorry, I cannot help with that.";

const QWEN_XML_PARIS_WEATHER_CALL: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

fn negotiate_with_cat_tools() -> Vec<ChatTool> {
    vec![ChatTool {
        name: "negotiate_with_cat".to_owned(),
        description: "Attempt to negotiate with a cat. Outcomes are not guaranteed and may include the silent treatment.".to_owned(),
        parameters: Some(json!({
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "What you are trying to negotiate, e.g. 'get off the keyboard' or 'stop knocking things off the table'"
                },
                "bribe": {
                    "type": "string",
                    "enum": ["tuna", "salmon", "treats", "ear_scritches", "cardboard_box", "none"],
                    "description": "What you are offering in exchange"
                },
                "desperation_level": {
                    "type": "integer",
                    "description": "How desperate you are, on a scale from 1 (mildly annoyed human) to 10 (it is 3am)",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["topic"],
            "additionalProperties": false
        })),
    }]
}

fn assert_parses_to_paris_weather_call(fixture: &LlamaFixture<'_>, answer: &str) -> Result<()> {
    let session =
        user_turn_chat_session(fixture.model, WEATHER_QUESTION, &get_weather_tools(), false)?;
    let parsed = recognized_answer(&session, answer, ChatOutputCompleteness::Finished);

    assert_eq!(parsed.tool_calls, [paris_weather_tool_call()]);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn gemma4_parses_its_tool_call_format(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_parses_to_paris_weather_call(
        fixture,
        "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}<tool_call|>",
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn glm47_parses_its_tool_call_format(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_parses_to_paris_weather_call(
        fixture,
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/Ministral-3-14B-Reasoning-2512-GGUF", "Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn mistral3_parses_its_tool_call_format(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_parses_to_paris_weather_call(
        fixture,
        r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#,
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_its_tool_call_format(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_parses_to_paris_weather_call(fixture, QWEN_XML_PARIS_WEATHER_CALL)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_tool_call_arguments_into_their_schema_types(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        "Get the cat off the keyboard.",
        &negotiate_with_cat_tools(),
        false,
    )?;
    let parsed = recognized_answer(
        &session,
        "<tool_call>\n\
<function=negotiate_with_cat>\n\
<parameter=topic>\n\
get off the keyboard\n\
</parameter>\n\
<parameter=bribe>\n\
tuna\n\
</parameter>\n\
<parameter=desperation_level>\n\
8\n\
</parameter>\n\
</function>\n\
</tool_call>",
        ChatOutputCompleteness::Finished,
    );

    assert_eq!(
        parsed.tool_calls,
        [ParsedToolCall::new(
            ParsedToolCallId::NotProvidedByModel,
            "negotiate_with_cat".to_owned(),
            ToolCallArguments::ValidJson(json!({
                "bribe": "tuna",
                "desperation_level": 8,
                "topic": "get off the keyboard",
            })),
        )]
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
)]
fn qwen35_parses_a_tool_call_cut_inside_its_arguments_as_a_named_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session =
        user_turn_chat_session(fixture.model, WEATHER_QUESTION, &get_weather_tools(), false)?;
    let parsed = recognized_answer(
        &session,
        "<tool_call>\n<function=get_weather>\n<parameter=lo",
        ChatOutputCompleteness::Partial,
    );

    assert_eq!(
        parsed
            .tool_calls
            .iter()
            .map(|tool_call| tool_call.name.as_str())
            .collect::<Vec<&str>>(),
        ["get_weather"]
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
)]
fn qwen35_parses_consecutive_tool_calls_in_order(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        fixture.model.chat_templates()?,
        &[user_message("What is the weather in Paris and in Berlin?")],
        &get_weather_tools(),
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: false,
            parallel_tool_calls: true,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;
    let parsed = recognized_answer(
        &session,
        &format!(
            "{QWEN_XML_PARIS_WEATHER_CALL}\n{}",
            QWEN_XML_PARIS_WEATHER_CALL.replace("Paris", "Berlin")
        ),
        ChatOutputCompleteness::Finished,
    );

    assert_eq!(
        parsed.tool_calls,
        [
            paris_weather_tool_call(),
            ParsedToolCall::new(
                ParsedToolCallId::NotProvidedByModel,
                "get_weather".to_owned(),
                ToolCallArguments::ValidJson(json!({"location": "Berlin"})),
            ),
        ]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn plain_content_answering_a_tool_request_parses_without_tool_calls(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session =
        user_turn_chat_session(fixture.model, WEATHER_QUESTION, &get_weather_tools(), false)?;
    let parsed = recognized_answer(
        &session,
        PLAIN_CONTENT_REFUSAL,
        ChatOutputCompleteness::Finished,
    );

    assert_eq!(parsed.tool_calls, []);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_plain_content_without_tools_parses_without_tool_calls(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", &[], false)?;
    let parsed = recognized_answer(&session, "Hello there.", ChatOutputCompleteness::Finished);

    assert_eq!(parsed.tool_calls, []);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_reports_a_finished_tool_call_its_parser_stops_short_of_with_the_format_it_expected(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session =
        user_turn_chat_session(fixture.model, WEATHER_QUESTION, &get_weather_tools(), false)?;

    assert_eq!(
        session
            .read(
                &ChatOutput::from(MALFORMED_QWEN_XML_TOOL_CALL),
                ChatOutputCompleteness::Finished
            )?
            .parse(),
        Ok(&ChatMessageParseOutcome::Unrecognized(RawChatMessage {
            text: MALFORMED_QWEN_XML_TOOL_CALL.to_owned(),
            completeness: ChatOutputCompleteness::Finished,
            expected_format: "peg-native".to_owned(),
        }))
    );

    Ok(())
}
