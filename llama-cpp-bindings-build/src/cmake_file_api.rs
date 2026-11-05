use std::fs;
use std::fs::DirEntry;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use serde::de::DeserializeOwned;

use crate::build_error::BuildError;
use crate::cmake_cache::CmakeCache;
use crate::cmake_reply_index::CmakeReplyIndex;

const REPLY_INDEX_PREFIX: &[u8] = b"index-";

fn read_reply_file<TReply: DeserializeOwned>(path: PathBuf) -> Result<TReply, BuildError> {
    let contents =
        fs::read(&path).map_err(|source| BuildError::CmakeFileApiReplyFileUnreadable {
            path: path.clone(),
            source,
        })?;

    serde_json::from_slice(&contents)
        .map_err(|source| BuildError::CmakeFileApiReplyFileInvalid { path, source })
}

fn current_reply_index(reply_dir: &Path) -> Result<PathBuf, BuildError> {
    fs::read_dir(reply_dir)
        .and_then(Iterator::collect::<Result<Vec<DirEntry>, io::Error>>)
        .map_err(|source| BuildError::CmakeFileApiReplyDirectoryUnreadable {
            path: reply_dir.to_path_buf(),
            source,
        })?
        .into_iter()
        .map(|entry| entry.file_name())
        .filter(|file_name| file_name.as_encoded_bytes().starts_with(REPLY_INDEX_PREFIX))
        .max()
        .map(|file_name| reply_dir.join(file_name))
        .ok_or_else(|| BuildError::CmakeFileApiReplyIndexMissing {
            path: reply_dir.to_path_buf(),
        })
}

/// The `CMake` file API of the build tree in `build_dir`, through which the build learns what
/// `CMake` resolved while configuring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CmakeFileApi {
    pub build_dir: PathBuf,
}

impl CmakeFileApi {
    /// The initial-cache script that, passed to `CMake` with `-C`, has every configure of the
    /// build tree query its cache before `CMake` reads the queries.
    pub const CACHE_QUERY_SCRIPT: &'static str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/cmake/cache_query.cmake");

    /// The cache `CMake` reported in its current reply to [`Self::CACHE_QUERY_SCRIPT`].
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::CmakeFileApiReplyDirectoryUnreadable`] when `CMake` wrote no
    /// reply, [`BuildError::CmakeFileApiReplyIndexMissing`] when the reply has no index, or
    /// [`BuildError::CmakeFileApiReplyFileUnreadable`] and
    /// [`BuildError::CmakeFileApiReplyFileInvalid`] when a reply file cannot be read as the
    /// reply this crate queried.
    pub fn cache(&self) -> Result<CmakeCache, BuildError> {
        let reply_dir = self.build_dir.join(".cmake/api/v1/reply");
        let CmakeReplyIndex { reply } = read_reply_file(current_reply_index(&reply_dir)?)?;

        read_reply_file(reply_dir.join(reply.cache.json_file))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::CmakeFileApi;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cmake_file_api");

    fn cache_failure(build_tree: &str) -> Option<String> {
        CmakeFileApi {
            build_dir: PathBuf::from(FIXTURES).join(build_tree),
        }
        .cache()
        .err()
        .map(|error| error.to_string())
    }

    #[test]
    fn a_build_tree_cmake_never_replied_to_is_refused() {
        assert_eq!(
            cache_failure("never_replied"),
            Some(format!(
                "the CMake file API reply directory {FIXTURES}/never_replied/.cmake/api/v1/reply \
                 could not be listed: No such file or directory (os error 2)"
            ))
        );
    }

    #[test]
    fn a_reply_without_an_index_is_refused() {
        assert_eq!(
            cache_failure("reply_without_index"),
            Some(format!(
                "CMake wrote no file API reply index into \
                 {FIXTURES}/reply_without_index/.cmake/api/v1/reply"
            ))
        );
    }

    #[test]
    fn an_index_naming_a_missing_reply_file_is_refused() {
        assert_eq!(
            cache_failure("index_naming_a_missing_cache"),
            Some(format!(
                "the CMake file API reply file \
                 {FIXTURES}/index_naming_a_missing_cache/.cmake/api/v1/reply/cache-v2-missing.json \
                 could not be read: No such file or directory (os error 2)"
            ))
        );
    }

    #[test]
    fn an_index_without_a_cache_reply_is_refused() {
        assert_eq!(
            cache_failure("index_without_cache_reply"),
            Some(format!(
                "the CMake file API reply file \
                 {FIXTURES}/index_without_cache_reply/.cmake/api/v1/reply/index-0.json is not the \
                 reply this crate queried: missing field `cache-v2` at line 1 column 11"
            ))
        );
    }
}
