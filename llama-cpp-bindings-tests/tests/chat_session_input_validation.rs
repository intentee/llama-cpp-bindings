use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatSessionError;
use llama_cpp_bindings::NewLlamaChatMessageError;
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
