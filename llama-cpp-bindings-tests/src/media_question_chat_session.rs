use anyhow::Result;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::mtmd_default_marker::mtmd_default_marker;

use crate::user_turn_chat_session::user_turn_chat_session;

/// A chat session for one user turn that shows a medium and asks `question` about it, with
/// thinking enabled and no tools.
///
/// # Errors
///
/// Forwards message construction and chat session errors.
///
/// # Panics
///
/// Panics when llama.cpp's media marker is not UTF-8.
pub fn media_question_chat_session<'model>(
    model: &'model LlamaModel,
    question: &str,
) -> Result<ChatSession<'model>> {
    user_turn_chat_session(
        model,
        &format!(
            "{}{question}",
            mtmd_default_marker().expect("llama.cpp's media marker must be UTF-8")
        ),
        "[]",
        true,
    )
}
