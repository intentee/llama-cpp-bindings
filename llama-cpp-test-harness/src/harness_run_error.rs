use thiserror::Error;

use llama_cpp_bindings::error::llama_cpp_error::LlamaCppError;

#[derive(Debug, Eq, Error, PartialEq)]
pub enum HarnessRunError {
    #[error(
        "the test harness requires --test-threads=1 (or unset); got --test-threads={requested}"
    )]
    ConflictingTestThreads { requested: usize },
    #[error(
        "two trials are named `{name}`; every trial needs its own name so its result can be told \
         apart"
    )]
    DuplicateTrialName { name: &'static str },
    #[error("failed to initialise the llama backend: {0}")]
    BackendInit(#[from] LlamaCppError),
}
