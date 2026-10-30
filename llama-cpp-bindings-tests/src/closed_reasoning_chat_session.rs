use anyhow::Result;
use llama_cpp_bindings::ChatAssistantPrefill;
use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::LlamaModel;

const PREFILLED_REASONING: &str = "\nThe question is simple, so I answer it directly.\n";
const PREFILLED_CONTENT: &str = "\n\n";

/// A chat session for one user turn whose answer continues prefilled content.
///
/// The prefill closes a reasoning block before the content begins, so the model writes only
/// content even when its template cannot disable thinking.
///
/// # Errors
///
/// Forwards message construction and chat session errors.
pub fn closed_reasoning_chat_session<'model>(
    model: &'model LlamaModel,
    question: &str,
    tools_json: &str,
) -> Result<ChatSession<'model>> {
    Ok(ChatSession::new(
        model,
        &[LlamaChatMessage::new(
            "user".to_owned(),
            question.to_owned(),
        )?],
        tools_json,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
                reasoning_content: PREFILLED_REASONING.to_owned(),
                content: PREFILLED_CONTENT.to_owned(),
            }),
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?)
}
