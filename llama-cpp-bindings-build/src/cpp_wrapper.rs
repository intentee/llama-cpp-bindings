use std::path::Path;
use std::path::PathBuf;

use crate::BuildContext;
use crate::cpp_build::cpp_build;
use crate::glob_paths;

const WRAPPER_SOURCE_PATTERNS: &[&str] = &["wrapper_*.cpp"];

fn wrapper_include_dirs(wrapper_dir: &Path, llama_src: &Path) -> Vec<PathBuf> {
    vec![
        wrapper_dir.to_path_buf(),
        wrapper_dir.join("GSL/include"),
        llama_src.to_path_buf(),
        llama_src.join("common"),
        llama_src.join("include"),
        llama_src.join("ggml/include"),
        llama_src.join("vendor"),
    ]
}

fn wrapper_sources(wrapper_dir: &Path) -> Vec<PathBuf> {
    let mut sources = Vec::new();

    for pattern in WRAPPER_SOURCE_PATTERNS {
        match glob_paths::collect_paths(wrapper_dir, pattern) {
            Ok(paths) => sources.extend(paths),
            Err(error) => panic!("cpp wrapper discovery failed: {error}"),
        }
    }

    sources
}

pub fn compile_cpp_wrappers(context: &BuildContext) {
    wrapper_build(context).compile("llama_cpp_bindings_sys_common_wrapper");
}

fn wrapper_build(context: &BuildContext) -> cc::Build {
    cpp_build(
        context,
        wrapper_include_dirs(&context.manifest_dir, &context.llama_src),
        wrapper_sources(&context.manifest_dir),
    )
}

#[cfg(test)]
mod tests {
    use crate::host_platform::HostPlatform;
    use crate::host_target_triple::host_target_triple;
    use crate::scratch_dir::ScratchDir;
    use crate::test_build_context::test_build_context;

    use super::compile_cpp_wrappers;
    use super::wrapper_include_dirs;
    use super::wrapper_sources;

    #[test]
    fn the_include_list_covers_the_wrapper_and_upstream_trees() {
        let dirs = wrapper_include_dirs(
            std::path::Path::new("/sys"),
            std::path::Path::new("/sys/llama.cpp"),
        );

        assert!(dirs.contains(&std::path::PathBuf::from("/sys")));
        assert!(dirs.contains(&std::path::PathBuf::from("/sys/GSL/include")));
        assert!(dirs.contains(&std::path::PathBuf::from("/sys/llama.cpp/ggml/include")));
    }

    #[test]
    fn only_wrapper_sources_are_discovered() {
        let scratch = ScratchDir::new("cppwrapper-sources");
        std::fs::write(scratch.path().join("wrapper_a.cpp"), b"").expect("source must be writable");
        std::fs::write(scratch.path().join("other.cpp"), b"").expect("source must be writable");

        let sources = wrapper_sources(scratch.path());

        assert_eq!(sources.len(), 1, "got: {sources:?}");
        assert!(sources[0].ends_with("wrapper_a.cpp"));
    }

    #[test]
    #[should_panic(expected = "cpp wrapper discovery failed")]
    fn a_directory_without_wrappers_panics() {
        let scratch = ScratchDir::new("cppwrapper-empty");

        let _ = wrapper_sources(scratch.path());
    }

    #[test]
    fn wrappers_are_compiled_into_an_archive() {
        let scratch = ScratchDir::new("cppwrapper-compile");
        std::fs::write(
            scratch.path().join("wrapper_probe.cpp"),
            b"extern \"C\" int llama_rs_probe() { return 7; }\n",
        )
        .expect("source must be writable");
        let context = test_build_context(
            scratch.path(),
            &scratch.path().join("llama.cpp"),
            &host_target_triple(),
        );

        compile_cpp_wrappers(&context);

        let archive = HostPlatform::current()
            .static_library_file_name("llama_cpp_bindings_sys_common_wrapper");

        assert!(
            scratch.path().join(&archive).exists(),
            "the wrapper archive {archive} must be produced"
        );
    }
}
