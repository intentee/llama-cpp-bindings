use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::path::PathBuf;

use hf_hub::api::sync::ApiError;

use crate::download_model::download_model;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GgufSource {
    HuggingFace {
        repo: &'static str,
        file: &'static str,
    },
    LocalPath(&'static str),
}

impl Display for GgufSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::HuggingFace { repo, file } => write!(formatter, "{repo} / {file}"),
            Self::LocalPath(path) => write!(formatter, "local:{path}"),
        }
    }
}

impl GgufSource {
    /// # Errors
    ///
    /// Returns the [`ApiError`] of downloading a `HuggingFace` source. `LocalPath` is infallible
    /// here; whether the file exists is checked when it is loaded.
    pub fn resolve_path(self) -> Result<PathBuf, ApiError> {
        match self {
            Self::HuggingFace { repo, file } => download_model(repo, file),
            Self::LocalPath(path) => Ok(PathBuf::from(path)),
        }
    }
}
