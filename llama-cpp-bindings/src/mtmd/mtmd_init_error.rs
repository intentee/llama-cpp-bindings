use std::path::PathBuf;

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum MtmdInitError {
    #[error(transparent)]
    FfiStatus(#[from] crate::FfiStatusError),
    #[error(transparent)]
    FfiContract(#[from] crate::FfiContractError),
    #[error("Failed to create CString from mmproj path: {0}")]
    CStringError(#[from] std::ffi::NulError),
    #[error("Mmproj path is not valid UTF-8: {0:?}")]
    PathToStrError(PathBuf),
    #[error("mmproj could not be loaded: {path:?}")]
    Unloadable { path: PathBuf },
    #[error("not enough memory")]
    NotEnoughMemory,
    #[error("the vendored library ran out of memory")]
    VendoredOutOfMemory,
    #[error("{message}")]
    Reported { message: String },
}
