use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_session::ChatSession;

use crate::recognized_answer::recognized_answer;

const REASONING_PROBE: &str = "x";

/// # Panics
///
/// Panics when the parser fails on or does not recognize the probe, or when the probe continues
/// neither the prompt's reasoning nor its content.
#[must_use]
pub fn prompted_reasoning(session: &ChatSession<'_>) -> String {
    let probed = recognized_answer(session, REASONING_PROBE, ChatOutputCompleteness::Partial);

    if probed.content == REASONING_PROBE && probed.reasoning_content.is_empty() {
        return String::new();
    }

    probed
        .reasoning_content
        .strip_suffix(REASONING_PROBE)
        .expect("the probe must continue either the prompt's reasoning or its content")
        .to_owned()
}
