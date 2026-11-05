use std::path::Path;

use crate::vendored_include_dirs::VENDORED_INCLUDE_DIRS;

/// The compiler flags that include every vendored header directory under `sys_dir` as a system
/// directory, so warnings in vendored headers do not surface while ours still do.
#[must_use]
pub fn system_include_flags(sys_dir: &Path) -> Vec<String> {
    VENDORED_INCLUDE_DIRS
        .iter()
        .map(|include_dir| format!("-isystem{}", sys_dir.join(include_dir).display()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::system_include_flags;

    #[test]
    fn every_vendored_include_dir_becomes_a_system_include_under_the_sys_dir() {
        assert_eq!(
            system_include_flags(Path::new("/repo/llama-cpp-bindings-sys")),
            [
                "-isystem/repo/llama-cpp-bindings-sys/GSL/include",
                "-isystem/repo/llama-cpp-bindings-sys/llama.cpp",
                "-isystem/repo/llama-cpp-bindings-sys/llama.cpp/common",
                "-isystem/repo/llama-cpp-bindings-sys/llama.cpp/include",
                "-isystem/repo/llama-cpp-bindings-sys/llama.cpp/ggml/include",
                "-isystem/repo/llama-cpp-bindings-sys/llama.cpp/vendor",
            ]
        );
    }
}
