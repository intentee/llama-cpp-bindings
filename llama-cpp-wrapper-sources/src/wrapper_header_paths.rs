use std::path::Path;
use std::path::PathBuf;

use crate::constants::WRAPPER_HEADERS;

#[must_use]
pub fn wrapper_header_paths(sys_dir: &Path) -> Vec<PathBuf> {
    WRAPPER_HEADERS
        .iter()
        .map(|wrapper_header| sys_dir.join(wrapper_header))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use super::wrapper_header_paths;
    use crate::constants::WRAPPER_HEADERS;

    #[test]
    fn every_wrapper_header_is_resolved_against_the_sys_dir() {
        assert_eq!(
            wrapper_header_paths(Path::new("/repo/llama-cpp-bindings-sys")),
            WRAPPER_HEADERS
                .iter()
                .map(|header| PathBuf::from("/repo/llama-cpp-bindings-sys").join(header))
                .collect::<Vec<PathBuf>>()
        );
    }
}
