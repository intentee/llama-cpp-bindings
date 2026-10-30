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
/// # Panics
///
/// Panics when the question holds a NUL byte or the model's template cannot serve the request.
#[must_use]
pub fn closed_reasoning_chat_session<'model>(
    model: &'model LlamaModel,
    question: &str,
    tools_json: &str,
) -> ChatSession<'model> {
    ChatSession::new(
        model,
        &[
            LlamaChatMessage::new("user".to_owned(), question.to_owned())
                .expect("the question must hold no NUL byte"),
        ],
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
    )
    .expect("the model's template must serve a prefilled answer")
}
