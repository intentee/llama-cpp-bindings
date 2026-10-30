use crate::error::sequence_id_out_of_range::SequenceIdOutOfRange;
use crate::error::token_outside_vocabulary::TokenOutsideVocabulary;

#[derive(thiserror::Error, Clone, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    #[error("the batch already holds the {n_batch} tokens one decode of the context takes")]
    BatchFull { n_batch: u32 },
    #[error(transparent)]
    TokenOutsideVocabulary(#[from] TokenOutsideVocabulary),
    #[error("a token must belong to at least one sequence")]
    NoSequenceIds,
    #[error(transparent)]
    SequenceIdOutOfRange(#[from] SequenceIdOutOfRange),
    #[error("the batch already holds {n_tokens} tokens and one more would overflow i32")]
    TokenCountOverflow { n_tokens: i32 },
}
