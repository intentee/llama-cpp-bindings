use std::ffi::CString;
use std::ffi::NulError;
use std::num::TryFromIntError;
use std::path::PathBuf;
use std::str::Utf8Error;

use llama_cpp_ffi_status::ffi_error::FfiError;

use crate::mtmd_image_token_limit::MtmdImageTokenLimit;
use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtmdError {
    #[error("batch size {requested} exceeds context batch size {context_max}")]
    BatchSizeExceedsContextLimit { requested: i32, context_max: u32 },
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
    #[error("Bitmap-source path is not valid UTF-8: {0:?}")]
    BitmapPathNotUtf8(PathBuf),
    #[error("{message}")]
    BitmapReported { message: String },
    #[error("the llama.cpp library ran out of memory")]
    ChunkCopyLlamaCppOutOfMemory,
    #[error("not enough memory")]
    ChunkCopyNotEnoughMemory,
    #[error("{message}")]
    ChunkCopyReported { message: String },
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
    #[error(
        "a chunk fitted to decodes of {fitted_micro_batch_tokens} tokens cannot be evaluated in decodes of {evaluated_micro_batch_tokens}"
    )]
    EvaluatedMicroBatchSmallerThanFitted {
        evaluated_micro_batch_tokens: u32,
        fitted_micro_batch_tokens: u32,
    },
    #[error(transparent)]
    Ffi(#[from] FfiError),
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
        "llama.cpp returned 2: media preprocessing failed, or its tokenizer threw (text too large, missing vocab, media marker and bitmap count mismatch, or allocation failure)"
    )]
    MediaPreprocessingOrTokenizationFailed,
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
    #[error("Mmproj path is not valid UTF-8: {0:?}")]
    MmprojPathNotUtf8(PathBuf),
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
    #[error("the llama.cpp library ran out of memory")]
    TokenizeLlamaCppOutOfMemory,
    #[error("not enough memory")]
    TokenizeNotEnoughMemory,
    #[error("{message}")]
    TokenizeReported { message: String },
    #[error("mtmd_tokenize returned an unknown status code: {code}")]
    TokenizeUnknownStatus { code: i32 },
    #[error("Unknown MTMD input chunk type: {0}")]
    UnknownInputChunkType(llama_cpp_bindings_sys::mtmd_input_chunk_type),
}
