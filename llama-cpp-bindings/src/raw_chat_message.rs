use crate::chat_output_completeness::ChatOutputCompleteness;

#[derive(Debug, Eq, PartialEq)]
pub struct RawChatMessage {
    pub text: String,
    pub completeness: ChatOutputCompleteness,
    pub ffi_error_message: String,
}
