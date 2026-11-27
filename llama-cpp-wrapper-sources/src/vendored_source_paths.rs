use std::path::Path;
use std::path::PathBuf;

use crate::constants::VENDORED_SOURCES;

#[must_use]
pub fn vendored_source_paths(llama_cpp_dir: &Path) -> Vec<PathBuf> {
    VENDORED_SOURCES
        .iter()
        .map(|vendored_source| llama_cpp_dir.join(vendored_source))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use super::vendored_source_paths;

    #[test]
    fn every_vendored_source_is_resolved_against_the_llama_cpp_dir() {
        assert_eq!(
            vendored_source_paths(Path::new("/repo/llama-cpp-native/llama.cpp")),
            [
                PathBuf::from("/repo/llama-cpp-native/llama.cpp/tools/server/server-common.cpp"),
                PathBuf::from("/repo/llama-cpp-native/llama.cpp/tools/server/server-decision.cpp"),
            ]
        );
    }
}
