use std::ffi::CString;

use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::chat_session_error::ChatSessionError;
use llama_cpp_bindings::error::string_to_token_error::StringToTokenError;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;
use llama_cpp_bindings_tests::media_question_chat_session::media_question_chat_session;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 64,
    n_batch = 64,
    n_ubatch = 64,
)]
fn a_question_holding_a_nul_character_is_rendered_into_the_prompt_after_the_media_marker(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let prompt = media_question_chat_session(fixture.model, "describe\0this")?.prompt()?;

    assert!(
        prompt.contains(&format!("{}describe\0this", mtmd_default_marker()?)),
        "{prompt:?}"
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
fn tokenizing_a_prompt_whose_reasoning_prefill_holds_a_nul_character_is_refused(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = ChatSession::new(
        fixture.model,
        fixture.model.chat_templates()?,
        &[user_message("hello")],
        &[],
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
        session.tokenize_prompt(fixture.model).err(),
        Some(ChatSessionError::PromptTokenization(
            StringToTokenError::NulError(nul_error)
        ))
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
fn a_conversation_the_template_raises_on_is_refused_with_the_template_message(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let result = ChatSession::new(
        fixture.model,
        fixture.model.chat_templates()?,
        &[user_message("hello"), user_message("are you there?")],
        &[],
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: false,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    );

    assert_eq!(
        result.err(),
        Some(ChatSessionError::TemplateCannotServeRequest {
            message: "Unable to generate parser for this template. Automatic parser generation \
                      failed: \n------------\nWhile executing CallExpression at line 19, column \
                      27 in source:\n...% 2 == 0) -%}↵        {{ raise_exception(\"Conversation \
                      roles must alternate user...\n                                           \
                      ^\nError: Jinja Exception: Conversation roles must alternate \
                      user/assistant/user/assistant/..."
                .to_owned()
        })
    );

    Ok(())
}
