use llama_cpp_bindings::ChatOutput;
use llama_cpp_bindings::ChatOutputCompleteness;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ParsedChatMessage;

/// Parses `answer` with the session's parser.
///
/// # Panics
///
/// Panics when parsing fails or the parser does not recognize the answer.
#[must_use]
pub fn recognized_answer(
    session: &ChatSession<'_>,
    answer: &str,
    completeness: ChatOutputCompleteness,
) -> ParsedChatMessage {
    session
        .parse(&ChatOutput::from(answer), completeness)
        .expect("the session's parser must parse the answer")
        .into_recognized()
        .expect("the session's parser must recognize the answer")
}
