use anyhow::Result;
use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatOutputCompleteness;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::ParsedToolCall;
use llama_cpp_bindings::ParsedToolCallId;
use llama_cpp_bindings::ToolCallArguments;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings_tests::get_weather_tools_json::GET_WEATHER_TOOLS_JSON;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;
use serde_json::json;

const WEATHER_QUESTION: &str = "What is the weather in Paris?";
const PLAIN_CONTENT_REFUSAL: &str = "Sorry, I cannot help with that.";

const NEGOTIATE_WITH_CAT_TOOLS_JSON: &str = r#"[
    {
        "type": "function",
        "function": {
            "name": "negotiate_with_cat",
            "description": "Attempt to negotiate with a cat. Outcomes are not guaranteed and may include the silent treatment.",
            "parameters": {
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
            }
        }
    }
]"#;

const QWEN_XML_PARIS_WEATHER_CALL: &str = "<tool_call>\n\
<function=get_weather>\n\
<parameter=location>\n\
Paris\n\
</parameter>\n\
</function>\n\
</tool_call>";

fn assert_parses_to_paris_weather_call(fixture: &LlamaFixture<'_>, answer: &str) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        WEATHER_QUESTION,
        GET_WEATHER_TOOLS_JSON,
        false,
    )?;
    let parsed = recognized_answer(&session, answer, ChatOutputCompleteness::Finished)?;

    assert_eq!(parsed.tool_calls, [paris_weather_tool_call()]);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_K_M.gguf"),
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
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
    n_gpu_layers = 999,
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
        NEGOTIATE_WITH_CAT_TOOLS_JSON,
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
    )?;

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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_a_tool_call_cut_inside_its_arguments_as_a_named_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        WEATHER_QUESTION,
        GET_WEATHER_TOOLS_JSON,
        false,
    )?;
    let parsed = recognized_answer(
        &session,
        "<tool_call>\n<function=get_weather>\n<parameter=lo",
        ChatOutputCompleteness::Partial,
    )?;

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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_parses_consecutive_tool_calls_in_order(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            "What is the weather in Paris and in Berlin?".to_owned(),
        )?],
        GET_WEATHER_TOOLS_JSON,
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
    )?;

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
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn plain_content_answering_a_tool_request_parses_without_tool_calls(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        WEATHER_QUESTION,
        GET_WEATHER_TOOLS_JSON,
        false,
    )?;
    let parsed = recognized_answer(
        &session,
        PLAIN_CONTENT_REFUSAL,
        ChatOutputCompleteness::Finished,
    )?;

    assert_eq!(parsed.tool_calls, []);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_plain_content_without_tools_parses_without_tool_calls(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", "[]", false)?;
    let parsed = recognized_answer(&session, "Hello there.", ChatOutputCompleteness::Finished)?;

    assert_eq!(parsed.tool_calls, []);

    Ok(())
}
