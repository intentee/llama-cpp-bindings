use serde::Deserialize;

/// One entry of the `CMake` cache, as the file API's `cache` object lists it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct CmakeCacheEntry {
    pub name: String,
    pub value: String,
}
