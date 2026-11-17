use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::error::unknown_attention_type::UnknownAttentionType;
use crate::error::unknown_pooling_type::UnknownPoolingType;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaContextLoadError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("context could not be constructed")]
    Unconstructible,
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
    #[error(
        "{n_seq_max} sequences need an output each, but a batch holds at most {output_capacity} outputs"
    )]
    SequencesExceedOutputCapacity {
        n_seq_max: u32,
        output_capacity: u32,
    },
    #[error(transparent)]
    UnknownAttentionType(#[from] UnknownAttentionType),
    #[error(transparent)]
    UnknownPoolingType(#[from] UnknownPoolingType),
}
