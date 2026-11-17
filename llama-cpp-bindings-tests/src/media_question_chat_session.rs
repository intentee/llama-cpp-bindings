use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_default_marker::mtmd_default_marker;

use crate::user_turn_chat_session::user_turn_chat_session;

/// A chat session for one user turn that shows a medium and asks `question` about it, with
/// thinking enabled and no tools.
///
/// # Errors
///
/// Fails when llama.cpp's media marker is not UTF-8 or the chat session cannot be created.
pub fn media_question_chat_session(model: &LlamaModel, question: &str) -> Result<ChatSession> {
    let marker = mtmd_default_marker().context("llama.cpp's media marker must be UTF-8")?;

    user_turn_chat_session(model, &format!("{marker}{question}"), &[], true)
}
