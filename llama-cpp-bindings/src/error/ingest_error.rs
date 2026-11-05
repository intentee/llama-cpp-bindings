use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::error::token_to_string_error::TokenToStringError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum IngestError {
    #[error("the generated token could not be detokenized: {0}")]
    Detokenize(#[from] TokenToStringError),
    #[error("the chat template's parser could not classify the answer: {0}")]
    Classify(#[from] ParseChatMessageError),
}
