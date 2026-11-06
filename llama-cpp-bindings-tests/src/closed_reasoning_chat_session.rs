use llama_cpp_bindings::chat_assistant_prefill::ChatAssistantPrefill;
use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings_types::chat_tool::ChatTool;

use crate::user_message::user_message;

const PREFILLED_REASONING: &str = "\nThe question is simple, so I answer it directly.\n";
const PREFILLED_CONTENT: &str = "\n\n";

/// A chat session for one user turn whose answer continues prefilled content.
///
/// The prefill closes a reasoning block before the content begins, so the model writes only
/// content even when its template cannot disable thinking.
///
/// # Panics
///
/// Panics when the model's template cannot serve the request.
#[must_use]
pub fn closed_reasoning_chat_session(
    model: &LlamaModel,
    question: &str,
    tools: &[ChatTool],
) -> ChatSession {
    ChatSession::new(
        model,
        model
            .chat_templates()
            .expect("the model must have chat templates"),
        &[user_message(question)],
        tools,
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::ContinueContent(ChatAssistantPrefill {
                reasoning_content: PREFILLED_REASONING.to_owned(),
                content: PREFILLED_CONTENT.to_owned(),
            }),
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )
    .expect("the model's template must serve a prefilled answer")
}
