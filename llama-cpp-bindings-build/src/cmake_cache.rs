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

    /// The switch `name`, which the project writes into the cache as `ON` or `OFF`.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::CmakeCacheEntryMissing`] when the cache has no entry `name`, or
    /// [`BuildError::CmakeCacheEntryNotASwitch`] when it holds anything but `ON` or `OFF`.
    pub fn switch(&self, name: &str) -> Result<bool, BuildError> {
        match self.value(name)? {
            "ON" => Ok(true),
            "OFF" => Ok(false),
            value => Err(BuildError::CmakeCacheEntryNotASwitch {
                name: name.to_owned(),
                value: value.to_owned(),
            }),
        }
    }

    /// The elements of the `CMake` list `name`, which `CMake` separates with semicolons and
    /// writes as an empty value when the list has none.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::CmakeCacheEntryMissing`] when the cache has no entry `name`.
    pub fn list(&self, name: &str) -> Result<Vec<&str>, BuildError> {
        Ok(self
            .value(name)?
            .split(';')
            .filter(|element| !element.is_empty())
            .collect())
    }
}

#[cfg(test)]
pub mod tests {
    use super::CmakeCache;
    use crate::cmake_cache_entry::CmakeCacheEntry;

    pub fn entry(name: &str, value: &str) -> CmakeCacheEntry {
        CmakeCacheEntry {
            name: name.to_owned(),
            value: value.to_owned(),
        }
    }

    #[test]
    fn a_switch_written_as_anything_but_on_or_off_is_refused() {
        assert_eq!(
            CmakeCache {
                entries: vec![entry("GGML_OPENMP_ENABLED", "TRUE")],
            }
            .switch("GGML_OPENMP_ENABLED")
            .err()
            .map(|error| error.to_string()),
            Some(
                "the CMake cache entry GGML_OPENMP_ENABLED is TRUE, neither ON nor OFF".to_owned()
            )
        );
    }

    #[test]
    fn an_empty_list_has_no_elements() {
        assert_eq!(
            CmakeCache {
                entries: vec![entry("OpenMP_C_LIB_NAMES", "")],
            }
            .list("OpenMP_C_LIB_NAMES")
            .unwrap(),
            Vec::<&str>::new()
        );
    }
}
