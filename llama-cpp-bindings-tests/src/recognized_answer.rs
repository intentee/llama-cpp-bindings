use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::ChatMessageParseOutcome;
use llama_cpp_bindings::ChatOutput;
use llama_cpp_bindings::ChatOutputCompleteness;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ParsedChatMessage;

/// Parses `answer` with the session's parser, failing when the parser does not recognize it.
///
/// # Errors
///
/// Forwards parse errors, and fails when the answer does not match the session's format.
pub fn recognized_answer(
    session: &ChatSession<'_>,
    answer: &str,
    completeness: ChatOutputCompleteness,
) -> Result<ParsedChatMessage> {
    match session.parse(&ChatOutput::from(answer), completeness)? {
        ChatMessageParseOutcome::Recognized(parsed) => Ok(parsed),
        ChatMessageParseOutcome::Unrecognized(raw) => {
            bail!("the session's parser must recognize {answer:?}; got {raw:?}")
        }
    }
}
