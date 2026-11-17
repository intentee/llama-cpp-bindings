use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;

/// Parses `answer` with the session's parser.
///
/// # Errors
///
/// Fails when parsing fails or the parser does not recognize the answer.
pub fn recognized_answer(
    session: &ChatSession,
    answer: &str,
    completeness: ChatOutputCompleteness,
) -> Result<ParsedChatMessage> {
    let parse_outcome = session
        .read(&ChatOutput::from(answer), completeness)
        .context("the session's parser must read the answer")?
        .parse()
        .cloned()
        .map_err(Clone::clone)
        .context("the session's parser must parse the answer")?;

    ensure!(
        matches!(parse_outcome, ChatMessageParseOutcome::Recognized(_)),
        "the session's parser must recognize the answer: {parse_outcome:?}"
    );

    parse_outcome
        .into_recognized()
        .ok()
        .context("the session's parser must recognize the answer")
}
