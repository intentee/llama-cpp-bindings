use llama_cpp_model::llama_token::LlamaToken;

use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;

#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    #[error("the batch already holds the {n_batch} tokens one decode of the context takes")]
    BatchFull { n_batch: u32 },
    #[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
    TokenOutsideVocabulary { token: LlamaToken, n_vocab: i32 },
    #[error("a token must belong to at least one sequence")]
    NoSequenceIds,
    #[error(transparent)]
    SequenceIdOutOfRange(#[from] SequenceIdOutOfRange),
    #[error("the batch already holds {n_tokens} tokens and one more would overflow i32")]
    TokenCountOverflow { n_tokens: i32 },
    #[error("token index {token_index} is not one of the {n_tokens} tokens the batch holds")]
    TokenIndexOutsideBatch { token_index: i32, n_tokens: i32 },
}
