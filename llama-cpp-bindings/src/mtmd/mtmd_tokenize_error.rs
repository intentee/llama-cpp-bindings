use crate::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdTokenizeError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("Failed to create CString from input text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("{0}")]
    InputChunksError(#[from] MtmdInputChunksError),
    #[error("number of bitmaps does not match number of markers in the text")]
    BitmapCountDoesNotMatchMarkerCount,
    #[error(
        "llama.cpp returned 2: media preprocessing failed, or its tokenizer threw (text too large, missing vocab, media marker and bitmap count mismatch, or allocation failure)"
    )]
    MediaPreprocessingOrTokenizationFailed,
    #[error("mtmd_tokenize returned an unknown status code: {code}")]
    UnknownStatus { code: i32 },
    #[error("mtmd_tokenize_from_parts returned an unknown status code: {code}")]
    UnknownTokenizeFromPartsStatus { code: i32 },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
