use std::num::TryFromIntError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    #[error("Empty buffer")]
    EmptyBuffer,
    #[error(
        "a token in {sequence_ids} sequences does not fit a batch created for at most {n_seq_max} sequences per token"
    )]
    TooManySequenceIds { sequence_ids: i32, n_seq_max: i32 },
    #[error("the batch already holds {n_tokens} tokens and one more would overflow i32")]
    TokenCountOverflow { n_tokens: i32 },
    #[error("{value_description} does not fit into {target_type}")]
    IntegerOverflow {
        value_description: &'static str,
        target_type: &'static str,
        #[source]
        source: TryFromIntError,
    },
}
