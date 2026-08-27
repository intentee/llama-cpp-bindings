use std::num::TryFromIntError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum BatchAddError {
    #[error("Insufficient Space of {0}")]
    InsufficientSpace(usize),
    #[error("Empty buffer")]
    EmptyBuffer,
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
