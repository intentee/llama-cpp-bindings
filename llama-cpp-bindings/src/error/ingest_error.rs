use llama_cpp_model::model_error::ModelError;

use crate::error::parse_chat_message_error::ParseChatMessageError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum IngestError {
    #[error("the generated token could not be detokenized")]
    Detokenize(#[from] ModelError),
    #[error("the chat template's parser could not classify the answer: {0}")]
    Classify(#[from] ParseChatMessageError),
}
