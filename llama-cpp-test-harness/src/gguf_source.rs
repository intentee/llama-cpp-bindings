use std::path::PathBuf;

use anyhow::Result;

use crate::download_model::download_model;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GgufSource {
    HuggingFace {
        repo: &'static str,
        file: &'static str,
    },
    LocalPath(&'static str),
}

impl GgufSource {
    /// # Errors
    ///
    /// Returns an error if the HF download fails. `LocalPath` is infallible here — file
    /// existence is checked when the file is loaded.
    pub fn resolve_path(self) -> Result<PathBuf> {
        match self {
            Self::HuggingFace { repo, file } => download_model(repo, file),
            Self::LocalPath(path) => Ok(PathBuf::from(path)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::GgufSource;

    #[test]
    fn resolve_path_for_local_path_returns_the_literal_path() {
        let source = GgufSource::LocalPath("/abs/example.gguf");

        let resolved = source
            .resolve_path()
            .expect("LocalPath resolve is infallible");

        assert_eq!(resolved, PathBuf::from("/abs/example.gguf"));
    }
}
