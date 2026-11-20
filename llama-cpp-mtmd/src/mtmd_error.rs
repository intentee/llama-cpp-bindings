use std::ffi::CString;
use std::ffi::NulError;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::str::Utf8Error;
use std::string::FromUtf8Error;

use llama_cpp_bindings_sys::exception_message_error::ExceptionMessageError;

use crate::mtmd_image_token_limit::MtmdImageTokenLimit;
use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtmdError {
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit { requested: i32, context_max: u32 },
    #[error(
        "batch size {requested} exceeds the micro batch {micro_batch} of a context that decodes \
         every batch in a single micro batch"
    )]
    BatchSizeExceedsSingleMicroBatch { requested: i32, micro_batch: u32 },
    #[error("number of bitmaps does not match number of markers in the text")]
    BitmapCountDoesNotMatchMarkerCount,
    #[error("bitmap data could not be decoded")]
    BitmapDecodeFailed,
    #[error("the llama.cpp library ran out of memory")]
    BitmapLlamaCppOutOfMemory,
    #[error("not enough memory")]
    BitmapNotEnoughMemory,
    #[error("Failed to create CString from bitmap-source path")]
    BitmapPathContainsNul(#[source] NulError),
    #[error("{message}")]
    BitmapReported { message: String },
    #[error("the llama.cpp library ran out of memory")]
    ChunkCopyLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ChunkCopyNotEnoughMemory,
    #[error("{message}")]
    ChunkCopyReported { message: String },
    #[error(
        "a chunk of {n_tokens} tokens over {n_positions} positions, evaluated from position \
         {start_position}, runs past the last position llama.cpp can hold"
    )]
    ChunkPositionsOverflow {
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        n_tokens: usize,
        n_positions: llama_cpp_bindings_sys::bindings::llama_pos,
    },
    #[error(
        "the llama context was created for another model than the one the multimodal projector \
         was loaded for"
    )]
    ContextModelIsNotTheProjectorModel,
    #[error("llama.cpp mtmd_default_marker returned bytes that are not valid UTF-8")]
    DefaultMarkerNotUtf8(#[source] Utf8Error),
    #[error("the llama.cpp library ran out of memory")]
    EncodeLlamaCppOutOfMemory,
    #[error("not enough memory")]
    EncodeNotEnoughMemory,
    #[error("{message}")]
    EncodeReported { message: String },
    #[error("multimodal chunk encoding failed with code: {code}")]
    EncodingFailed { code: i32 },
    #[error("multimodal chunk eval failed with code: {code}")]
    EvalFailed { code: i32 },
    #[error("the llama.cpp library ran out of memory")]
    EvalLlamaCppOutOfMemory,
    #[error("not enough memory")]
    EvalNotEnoughMemory,
    #[error("{message}")]
    EvalReported { message: String },
    #[error("the chunk type could not be classified before evaluating it")]
    EvalUnknownChunkType(#[source] Box<Self>),
    #[error(
        "a chunk fitted to decodes of {fitted_micro_batch_tokens} tokens cannot be evaluated in decodes of {evaluated_micro_batch_tokens}"
    )]
    EvaluatedMicroBatchSmallerThanFitted {
        evaluated_micro_batch_tokens: u32,
        fitted_micro_batch_tokens: u32,
    },
    #[error("the exception message reported by {operation} is not valid UTF-8")]
    ExceptionMessageNotUtf8 {
        operation: &'static str,
        #[source]
        source: FromUtf8Error,
    },
    #[error("bitmap file is unreadable: {path:?}")]
    FileUnreadable { path: PathBuf },
    #[error("Image dimensions {nx}x{ny} overflow the addressable RGB buffer size")]
    ImageDimensionsOverflow { nx: u32, ny: u32 },
    #[error("Image dimensions too small: {nx}x{ny} (minimum 2x2)")]
    ImageDimensionsTooSmall { nx: u32, ny: u32 },
    #[error("the image maximum token limit {limit:?} exceeds what a C int holds")]
    ImageMaxTokensExceedCInt {
        limit: MtmdImageTokenLimit,
        #[source]
        source: TryFromIntError,
    },
    #[error("the image minimum token limit {limit:?} exceeds what a C int holds")]
    ImageMinTokensExceedCInt {
        limit: MtmdImageTokenLimit,
        #[source]
        source: TryFromIntError,
    },
    #[error(transparent)]
    InitContextParams(Box<Self>),
    #[error("the llama.cpp library ran out of memory")]
    InitLlamaCppOutOfMemory,
    #[error("not enough memory")]
    InitNotEnoughMemory,
    #[error("{message}")]
    InitReported { message: String },
    #[error("the llama.cpp library ran out of memory")]
    InputChunksLlamaCppOutOfMemory,
    #[error("not enough memory")]
    InputChunksNotEnoughMemory,
    #[error("{message}")]
    InputChunksReported { message: String },
    #[error("Failed to create CString from input text")]
    InputTextContainsNul(#[source] NulError),
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    #[error(
        "llama.cpp reported a malformed part list in {operation}: a part holding both or neither of text and bitmap, a text part with a null text pointer, or bitmaps that do not line up with their parts"
    )]
    LlamaCppReportedMalformedPartList { operation: &'static str },
    #[error(
        "llama.cpp returned 2: media preprocessing failed, or its tokenizer threw (text too large, missing vocab, media marker and bitmap count mismatch, or allocation failure)"
    )]
    MediaPreprocessingOrTokenizationFailed,
    #[error(transparent)]
    MemoryUsageContextParams(Box<Self>),
    #[error("{operation} measured memory on more devices than the backend registers")]
    MemoryUsageCoversMoreDevicesThanRegistered { operation: &'static str },
    #[error("the llama.cpp library ran out of memory")]
    MemoryUsageLlamaCppOutOfMemory,
    #[error("not enough memory")]
    MemoryUsageNotEnoughMemory,
    #[error("{message}")]
    MemoryUsageReported { message: String },
    #[error("the memory usage of mmproj {path:?} could not be measured, because it does not load")]
    MemoryUsageUnloadable { path: CString },
    #[error("Failed to create CString from mmproj path")]
    MmprojPathContainsNul(#[source] NulError),
    #[error("mmproj could not be loaded: {path:?}")]
    MmprojUnloadable { path: PathBuf },
    #[error(
        "a chunk decoded non-causally has {} tokens but a single decode fits {}",
        .0.chunk_tokens,
        .0.micro_batch_tokens,
    )]
    NonCausalChunkExceedsMicroBatch(NonCausalChunkMicroBatchMismatch),
    #[error("batch size {requested} must be positive")]
    NonPositiveBatchSize { requested: i32 },
    #[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
    SequenceIdOutOfRange { seq_id: i32, n_seq_max: u32 },
    #[error("mtmd_tokenize_from_parts returned an unknown status code: {code}")]
    TokenizeFromPartsUnknownStatus { code: i32 },
    #[error(transparent)]
    TokenizeInputChunks(Box<Self>),
    #[error("the llama.cpp library ran out of memory")]
    TokenizeLlamaCppOutOfMemory,
    #[error("not enough memory")]
    TokenizeNotEnoughMemory,
    #[error("{message}")]
    TokenizeReported { message: String },
    #[error("mtmd_tokenize returned an unknown status code: {code}")]
    TokenizeUnknownStatus { code: i32 },
    #[error("Unknown MTMD input chunk type: {0}")]
    UnknownInputChunkType(llama_cpp_bindings_sys::bindings::mtmd_input_chunk_type),
}

impl ExceptionMessageError for MtmdError {
    fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
        Self::ExceptionMessageNotUtf8 { operation, source }
    }
}
