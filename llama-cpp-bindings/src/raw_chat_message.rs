use crate::chat_output_completeness::ChatOutputCompleteness;

/// An answer the template's parser does not recognize, with the name of the format llama.cpp
/// expected it in.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawChatMessage {
    pub text: String,
    pub completeness: ChatOutputCompleteness,
    pub expected_format: String,
}
