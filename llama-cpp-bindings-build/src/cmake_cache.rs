use serde::Deserialize;

use crate::build_error::BuildError;
use crate::cmake_cache_entry::CmakeCacheEntry;

/// The `CMake` cache of a configured build tree, read through the file API's `cache` object.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CmakeCache {
    pub entries: Vec<CmakeCacheEntry>,
}

impl CmakeCache {
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.iter().any(|entry| entry.name == name)
    }

    /// # Errors
    ///
    /// Returns [`BuildError::CmakeCacheEntryMissing`] when the cache has no entry `name`.
    pub fn value(&self, name: &str) -> Result<&str, BuildError> {
        self.entries
            .iter()
            .find(|entry| entry.name == name)
            .map(|entry| entry.value.as_str())
            .ok_or_else(|| BuildError::CmakeCacheEntryMissing {
                name: name.to_owned(),
            })
    }
}

#[cfg(test)]
pub mod tests {
    use super::CmakeCache;
    use crate::cmake_cache_entry::CmakeCacheEntry;

    pub fn cache_of(entries: &[(&str, &str)]) -> CmakeCache {
        CmakeCache {
            entries: entries
                .iter()
                .map(|(name, value)| CmakeCacheEntry {
                    name: (*name).to_owned(),
                    value: (*value).to_owned(),
                })
                .collect(),
        }
    }
}
