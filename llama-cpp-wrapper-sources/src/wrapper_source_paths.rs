use std::path::Path;
use std::path::PathBuf;

use crate::wrapper_sources::WRAPPER_SOURCES;

#[must_use]
pub fn wrapper_source_paths(sys_dir: &Path) -> Vec<PathBuf> {
    WRAPPER_SOURCES
        .iter()
        .map(|wrapper_source| sys_dir.join(wrapper_source))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use super::wrapper_source_paths;
    use crate::wrapper_sources::WRAPPER_SOURCES;

    #[test]
    fn every_wrapper_source_is_resolved_against_the_sys_dir() {
        assert_eq!(
            wrapper_source_paths(Path::new("/repo/llama-cpp-bindings-sys")),
            WRAPPER_SOURCES
                .iter()
                .map(|source| PathBuf::from("/repo/llama-cpp-bindings-sys").join(source))
                .collect::<Vec<PathBuf>>()
        );
    }
}
