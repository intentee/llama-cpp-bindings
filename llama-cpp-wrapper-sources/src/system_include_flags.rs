use std::path::Path;

use crate::constants::GSL_INCLUDE_DIR;
use crate::constants::LLAMA_CPP_INCLUDE_DIRS;

/// The compiler flags that include the vendored headers as system directories.
///
/// The GSL headers come from under `sys_dir` and the llama.cpp headers from under
/// `llama_cpp_dir`, so warnings in vendored headers do not surface while ours still do.
#[must_use]
pub fn system_include_flags(sys_dir: &Path, llama_cpp_dir: &Path) -> Vec<String> {
    [sys_dir.join(GSL_INCLUDE_DIR), llama_cpp_dir.to_path_buf()]
        .into_iter()
        .chain(
            LLAMA_CPP_INCLUDE_DIRS
                .iter()
                .map(|include_dir| llama_cpp_dir.join(include_dir)),
        )
        .map(|include_dir| format!("-isystem{}", include_dir.display()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::system_include_flags;

    #[test]
    fn gsl_and_every_llama_cpp_include_dir_become_system_includes_under_their_roots() {
        assert_eq!(
            system_include_flags(
                Path::new("/repo/llama-cpp-bindings-sys"),
                Path::new("/repo/llama-cpp-native/llama.cpp")
            ),
            [
                "-isystem/repo/llama-cpp-bindings-sys/GSL/include",
                "-isystem/repo/llama-cpp-native/llama.cpp",
                "-isystem/repo/llama-cpp-native/llama.cpp/common",
                "-isystem/repo/llama-cpp-native/llama.cpp/include",
                "-isystem/repo/llama-cpp-native/llama.cpp/ggml/include",
                "-isystem/repo/llama-cpp-native/llama.cpp/vendor",
            ]
        );
    }
}
