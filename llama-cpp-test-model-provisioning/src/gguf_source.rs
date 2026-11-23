use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::path::PathBuf;

use hf_hub::Repo;
use hf_hub::RepoType;
use hf_hub::api::sync::ApiError;

use crate::download_model::download_model;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GgufSource {
    HuggingFace {
        repo: &'static str,
        file: &'static str,
    },
    HuggingFaceShards {
        repo: &'static str,
        commit: &'static str,
        first_shard: &'static str,
        remaining_shards: &'static [&'static str],
    },
    LocalPath(&'static str),
}

impl Display for GgufSource {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::HuggingFace { repo, file } => write!(formatter, "{repo} / {file}"),
            Self::HuggingFaceShards {
                repo,
                commit,
                first_shard,
                remaining_shards,
            } => write!(
                formatter,
                "{repo}@{commit} / {first_shard} + {}",
                remaining_shards.join(" + ")
            ),
            Self::LocalPath(path) => write!(formatter, "local:{path}"),
        }
    }
}

impl GgufSource {
    /// Resolves the source to the local path llama.cpp loads it from. A sharded source downloads
    /// every shard of its commit into one snapshot directory, where llama.cpp finds the shards
    /// that follow the first one.
    ///
    /// # Errors
    ///
    /// Returns the [`ApiError`] of downloading a `HuggingFace` file or any shard of a
    /// `HuggingFaceShards` source. `LocalPath` is infallible here; whether the file exists is
    /// checked when it is loaded.
    pub fn resolve_path(self) -> Result<PathBuf, ApiError> {
        match self {
            Self::HuggingFace { repo, file } => download_model(&Repo::model(repo.to_owned()), file),
            Self::HuggingFaceShards {
                repo,
                commit,
                first_shard,
                remaining_shards,
            } => {
                let pinned_repo =
                    Repo::with_revision(repo.to_owned(), RepoType::Model, commit.to_owned());

                download_model(&pinned_repo, first_shard).and_then(|first_shard_path| {
                    remaining_shards
                        .iter()
                        .try_for_each(|shard| download_model(&pinned_repo, shard).map(drop))
                        .map(|()| first_shard_path)
                })
            }
            Self::LocalPath(path) => Ok(PathBuf::from(path)),
        }
    }
}
