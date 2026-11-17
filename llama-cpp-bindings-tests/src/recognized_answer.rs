use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_session::ChatSession;

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
    session
        .read(&ChatOutput::from(answer), completeness)
        .context("the session's parser must read the answer")
        .and_then(|reading| {
            reading
                .parse()
                .cloned()
                .map_err(Clone::clone)
                .context("the session's parser must parse the answer")
        })
        .and_then(|parse_outcome| {
            let unrecognized_answer =
                format!("the session's parser must recognize the answer: {parse_outcome:?}");

            parse_outcome.into_recognized().context(unrecognized_answer)
        })
}
