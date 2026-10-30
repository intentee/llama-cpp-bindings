use std::ffi::CString;

use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionError;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::NewLlamaChatMessageError;
use llama_cpp_bindings::StringToTokenError;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 64,
)]
fn media_question_chat_session_refuses_a_question_with_a_nul_byte(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let Err(failure) = media_question_chat_session(fixture.model, "describe\0this") else {
        bail!("an interior NUL byte in the question must refuse the message");
    };
    let NewLlamaChatMessageError::NulError(nul_error) =
        failure.downcast::<NewLlamaChatMessageError>()?;

    assert_eq!(nul_error.nul_position(), "<__media__>describe".len());

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
fn chat_session_refuses_a_tool_missing_its_function_name(fixture: &LlamaFixture<'_>) -> Result<()> {
    let tools_json = r#"[{"type":"function","function":{"description":"reports the weather"}}]"#;
    let result = user_turn_chat_session(fixture.model, "hello", tools_json, false);
    let Err(failure) = result else {
        bail!("a tool without a function name must refuse the chat session");
    };

    assert_eq!(
        failure.downcast::<ChatSessionError>()?,
        ChatSessionError::ToolsInvalid {
            message: "Failed to parse tools: [json.exception.out_of_range.403] key 'name' not \
                      found; tools = [\n  {\n    \"type\": \"function\",\n    \"function\": {\n      \
                      \"description\": \"reports the weather\"\n    }\n  }\n]"
                .to_owned(),
        }
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
fn chat_session_refuses_tools_with_a_nul_byte(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        user_turn_chat_session(fixture.model, "hello", "[\0]", false)
            .err()
            .and_then(|failure| failure.downcast::<ChatSessionError>().ok()),
        CString::new("[\0]")
            .err()
            .map(ChatSessionError::RequestContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn chat_session_refuses_a_message_that_is_not_utf8(fixture: &LlamaFixture<'_>) -> Result<()> {
    let not_utf8 = CString::new([0xff])?;
    let message = LlamaChatMessage {
        role: CString::new("user")?,
        content: not_utf8.clone(),
    };

    assert_eq!(
        ChatSession::new(
            fixture.model,
            &[message],
            "[]",
            &ChatSessionOptions {
                assistant_turn: ChatAssistantTurn::New,
                enable_thinking: false,
                parallel_tool_calls: false,
                tool_choice: ChatToolChoice::Auto,
            },
        )
        .err(),
        not_utf8
            .to_str()
            .err()
            .map(ChatSessionError::MessageNotUtf8)
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
fn tokenizing_a_prompt_whose_reasoning_prefill_holds_a_nul_character_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            "hello".to_owned(),
        )?],
        "[]",
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::ContinueReasoning("a\0b".to_owned()),
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;
    let Some(nul_error) = session
        .prompt_parts()?
        .iter()
        .find_map(|part| CString::new(part.text()).err())
    else {
        bail!("the rendered prompt must hold the reasoning prefill's NUL character");
    };

    assert_eq!(
        session.tokenize_prompt().err(),
        Some(ChatSessionError::PromptTokenization(
            StringToTokenError::NulError(nul_error)
        ))
    );

    Ok(())
}
