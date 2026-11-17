use std::path::PathBuf;

use llama_cpp_ffi_status::ffi_error::FfiError;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdBitmapError {
    #[error(transparent)]
    Ffi(#[from] FfiError),
    #[error("Failed to create CString from bitmap-source path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Bitmap-source path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    #[error("Image dimensions {nx}x{ny} overflow the addressable RGB buffer size")]
    ImageDimensionsOverflow { nx: u32, ny: u32 },
    #[error("Image dimensions too small: {nx}x{ny} (minimum 2x2)")]
    ImageDimensionsTooSmall { nx: u32, ny: u32 },
    #[error("bitmap data could not be decoded")]
    BitmapDecodeFailed,
    #[error("bitmap file is unreadable: {path:?}")]
    FileUnreadable { path: PathBuf },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the llama.cpp library ran out of memory")]
    LlamaCppOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
