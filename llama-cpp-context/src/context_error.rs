use std::ffi::c_int;
use std::num::NonZeroI32;
use std::num::TryFromIntError;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_params::llama_pooling_type::LlamaPoolingType;
use llama_cpp_params::params_error::ParamsError;

use crate::context_limits::ContextLimits;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextError {
    #[error(
        "llama_batch_ext_add_token returned the code {code}, which llama.cpp does not document"
    )]
    BatchAddTokenReturnedUndocumentedCode { code: i32 },
    #[error("the batch already holds the {n_batch} tokens one decode of the context takes")]
    BatchFull { n_batch: u32 },
    #[error("the batch was built for context limits {batch:?}, but the context has {context:?}")]
    BatchLimitsMismatch {
        batch: ContextLimits,
        context: ContextLimits,
    },
    #[error("the llama.cpp library ran out of memory")]
    BatchTransferLlamaCppOutOfMemory,
    #[error("not enough memory")]
    BatchTransferNotEnoughMemory,
    #[error("{message}")]
    BatchTransferReported { message: String },
    #[error("context could not be constructed")]
    ContextUnconstructible,
    #[error("the llama.cpp library ran out of memory")]
    CreateLlamaCppOutOfMemory,
    #[error("not enough memory")]
    CreateNotEnoughMemory,
    #[error("{message}")]
    CreateReported { message: String },
    #[error("decode aborted by callback")]
    DecodeAborted,
    #[error("decode batch of {n_tokens} tokens must fit a single micro batch of {n_ubatch} tokens")]
    DecodeBatchExceedsMicroBatch { n_tokens: i32, n_ubatch: u32 },
    #[error("decode batch is invalid (empty, output mismatch, or initialization failure)")]
    DecodeBatchInvalid,
    #[error("the batch could not be handed to llama.cpp")]
    DecodeBatchTransfer(#[source] Box<Self>),
    #[error("backend compute failed during decode")]
    DecodeComputeFailed,
    #[error("no KV cache slot was available")]
    DecodeNoKvCacheSlot,
    #[error("decode ran out of memory")]
    DecodeOutOfMemory,
    #[error(transparent)]
    DecodeProcess(Box<Self>),
    #[error("decode returned an unknown status code: {code}")]
    DecodeUnknownStatus { code: c_int },
    #[error("token {token_index} names no output row of the last decode")]
    EmbeddingTokenNotAnOutput { token_index: i32 },
    #[error("Embeddings weren't enabled in the context options")]
    EmbeddingsNotEnabled,
    #[error("encode batch of {n_tokens} tokens must fit a single micro batch of {n_ubatch} tokens")]
    EncodeBatchExceedsMicroBatch { n_tokens: i32, n_ubatch: u32 },
    #[error("encode batch is invalid (empty or initialization failure)")]
    EncodeBatchInvalid,
    #[error("the batch could not be handed to llama.cpp")]
    EncodeBatchTransfer(#[source] Box<Self>),
    #[error("backend compute failed during encode")]
    EncodeComputeFailed,
    #[error("model has no encoder")]
    EncodeModelHasNoEncoder,
    #[error("no KV cache slot was available")]
    EncodeNoKvCacheSlot,
    #[error("encode ran out of memory")]
    EncodeOutOfMemory,
    #[error(transparent)]
    EncodeProcess(Box<Self>),
    #[error("encode returned an unknown status code: {code}")]
    EncodeUnknownStatus { code: c_int },
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("the model graph of the last decode produces no logits")]
    GraphProducesNoLogits,
    #[error("Invalid embedding dimension")]
    InvalidEmbeddingDimension(#[source] TryFromIntError),
    #[error("{operation}: llama.cpp refused the decision order of an entry it had just added")]
    LlamaCppRefusedDecisionOrderOfAddedEntry { operation: &'static str },
    #[error("{operation}: llama.cpp refused the output flag of an entry it had just added")]
    LlamaCppRefusedOutputFlagOfAddedEntry { operation: &'static str },
    #[error("{operation}: llama.cpp refused the position of an entry it had just added")]
    LlamaCppRefusedPositionOfAddedEntry { operation: &'static str },
    #[error(
        "{operation}: llama.cpp refused an additional sequence id of an entry, although the sequence ids were validated against the context"
    )]
    LlamaCppRefusedValidatedAdditionalSequenceId { operation: &'static str },
    #[error(
        "{operation}: llama.cpp refused an entry as full, although the batch was validated against its limits"
    )]
    LlamaCppRefusedValidatedEntryAsFull { operation: &'static str },
    #[error(
        "{operation}: llama.cpp refused a sequence id, although it was validated against the context"
    )]
    LlamaCppRefusedValidatedSequenceId { operation: &'static str },
    #[error(
        "{operation}: llama.cpp refused a token, although it was validated against the vocabulary"
    )]
    LlamaCppRefusedValidatedToken { operation: &'static str },
    #[error("logit for token index {0} is not initialized")]
    LogitTokenNotInitialized(i32),
    #[error("logits data pointer is null")]
    LogitsNull,
    #[error("the LoRA adapter at position {position} was loaded into another model")]
    LoraAdapterBelongsToAnotherModel { position: usize },
    #[error("a token must belong to at least one sequence")]
    NoSequenceIds,
    #[error(transparent)]
    Params(#[from] ParamsError),
    #[error(
        "the last decode pooled embeddings with {pooling_type:?} pooling, which produces no logits"
    )]
    PooledEmbeddingsProduceNoLogits { pooling_type: LlamaPoolingType },
    #[error("the llama.cpp library ran out of memory")]
    ProcessLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ProcessNotEnoughMemory,
    #[error("{message}")]
    ProcessReported { message: String },
    #[error(
        "No sequence embedding exists for sequence {sequence_index}; it was not part of the last decoded batch"
    )]
    SequenceEmbeddingUnavailable { sequence_index: i32 },
    #[error(
        "Sequence embeddings need a pooling context, but this context uses LLAMA_POOLING_TYPE_NONE"
    )]
    SequenceEmbeddingsRequirePooling,
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error(
        "{n_seq_max} sequences need an output each, but a batch holds at most {output_capacity} outputs"
    )]
    SequencesExceedOutputCapacity {
        n_seq_max: u32,
        output_capacity: u32,
    },
    #[error("llama_set_adapters_lora failed with status {0}")]
    SetLoraAdaptersFailed(i32),
    #[error("the batch already holds {n_tokens} tokens and one more would overflow i32")]
    TokenCountOverflow { n_tokens: i32 },
    #[error(
        "Token embeddings need a context with LLAMA_POOLING_TYPE_NONE, but it pools with {pooling_type:?}"
    )]
    TokenEmbeddingsRequireNonePooling { pooling_type: LlamaPoolingType },
    #[error("token index {token_index} is not one of the {n_tokens} tokens the batch holds")]
    TokenIndexOutsideBatch { token_index: i32, n_tokens: i32 },
    #[error("token {token} is outside the vocabulary of {n_vocab} tokens")]
    TokenOutsideVocabulary { token: LlamaToken, n_vocab: i32 },
    #[error(
        "a token cannot be added at position {pos}: llama.cpp overflows the arithmetic it does on \
         the positions of a context above {max_position}"
    )]
    TokenPositionAboveMax {
        pos: llama_pos,
        max_position: llama_pos,
    },
    #[error(
        "a token cannot be added at the negative position {pos}: llama.cpp marks empty KV cache \
         cells with -1 and holds no other negative position"
    )]
    TokenPositionNegative { pos: llama_pos },
    #[error("n_vocab does not fit into usize")]
    VocabSizeOverflow(#[source] TryFromIntError),
}

impl ContextError {
    #[must_use]
    pub const fn from_add_token_return_code(operation: &'static str, return_code: i32) -> Self {
        match return_code {
            -1 => Self::LlamaCppRefusedValidatedEntryAsFull { operation },
            -2 => Self::LlamaCppRefusedValidatedToken { operation },
            -3 => Self::LlamaCppRefusedValidatedSequenceId { operation },
            code => Self::BatchAddTokenReturnedUndocumentedCode { code },
        }
    }

    #[must_use]
    pub const fn from_decode_return_code(return_code: NonZeroI32) -> Self {
        match return_code.get() {
            1 => Self::DecodeNoKvCacheSlot,
            2 => Self::DecodeAborted,
            -1 => Self::DecodeBatchInvalid,
            -2 => Self::DecodeOutOfMemory,
            -3 => Self::DecodeComputeFailed,
            code => Self::DecodeUnknownStatus { code },
        }
    }

    #[must_use]
    pub const fn from_encode_return_code(return_code: NonZeroI32) -> Self {
        match return_code.get() {
            1 => Self::EncodeNoKvCacheSlot,
            -1 => Self::EncodeBatchInvalid,
            -2 => Self::EncodeOutOfMemory,
            -3 => Self::EncodeComputeFailed,
            code => Self::EncodeUnknownStatus { code },
        }
    }
}

impl ExceptionMessageError for ContextError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroI32;

    use super::ContextError;

    #[test]
    fn decode_no_kv_cache_slot_maps_from_code_one() {
        let error =
            ContextError::from_decode_return_code(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, ContextError::DecodeNoKvCacheSlot);
    }

    #[test]
    fn decode_aborted_maps_from_code_two() {
        let error =
            ContextError::from_decode_return_code(NonZeroI32::new(2).expect("2 is non-zero"));

        assert_eq!(error, ContextError::DecodeAborted);
    }

    #[test]
    fn decode_batch_invalid_maps_from_code_negative_one() {
        let error =
            ContextError::from_decode_return_code(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, ContextError::DecodeBatchInvalid);
    }

    #[test]
    fn decode_allocation_and_compute_failures_map_from_codes_negative_two_and_three() {
        assert_eq!(
            [-2, -3].map(|code| ContextError::from_decode_return_code(
                NonZeroI32::new(code).expect("non-zero code")
            )),
            [
                ContextError::DecodeOutOfMemory,
                ContextError::DecodeComputeFailed
            ]
        );
    }

    #[test]
    fn unrecognized_decode_code_falls_through_to_unknown_status() {
        let error =
            ContextError::from_decode_return_code(NonZeroI32::new(42).expect("42 is non-zero"));

        assert_eq!(error, ContextError::DecodeUnknownStatus { code: 42 });
    }

    #[test]
    fn encode_no_kv_cache_slot_maps_from_code_one() {
        let error =
            ContextError::from_encode_return_code(NonZeroI32::new(1).expect("1 is non-zero"));

        assert_eq!(error, ContextError::EncodeNoKvCacheSlot);
    }

    #[test]
    fn encode_batch_invalid_maps_from_code_negative_one() {
        let error =
            ContextError::from_encode_return_code(NonZeroI32::new(-1).expect("-1 is non-zero"));

        assert_eq!(error, ContextError::EncodeBatchInvalid);
    }

    #[test]
    fn encode_allocation_and_compute_failures_map_from_codes_negative_two_and_three() {
        assert_eq!(
            [-2, -3].map(|code| ContextError::from_encode_return_code(
                NonZeroI32::new(code).expect("non-zero code")
            )),
            [
                ContextError::EncodeOutOfMemory,
                ContextError::EncodeComputeFailed
            ]
        );
    }

    #[test]
    fn unrecognized_encode_code_falls_through_to_unknown_status() {
        let error =
            ContextError::from_encode_return_code(NonZeroI32::new(99).expect("99 is non-zero"));

        assert_eq!(error, ContextError::EncodeUnknownStatus { code: 99 });
    }
}
