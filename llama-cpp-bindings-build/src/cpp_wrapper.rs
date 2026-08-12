use std::path::Path;
use std::path::PathBuf;

use crate::glob_paths;
use crate::target_os::TargetOs;

fn msvc_flags(target_os: &TargetOs) -> &'static [&'static str] {
    if target_os.is_msvc() {
        &["/std:c++17", "/EHsc"]
    } else {
        &[]
    }
}

fn links_stdlib_statically(target_os: &TargetOs) -> bool {
    target_os.is_android() && cfg!(feature = "static-stdcxx")
}

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
        let scoped = wrapper_dir.join(pattern);

        match glob_paths::collect_paths(&scoped.to_string_lossy()) {
            Ok(paths) => sources.extend(paths),
            Err(error) => panic!("cpp wrapper discovery failed: {error}"),
        }
    }

    sources
}

pub fn compile_cpp_wrappers(wrapper_dir: &Path, llama_src: &Path, target_os: &TargetOs) {
    wrapper_build(wrapper_dir, llama_src, target_os)
        .compile("llama_cpp_bindings_sys_common_wrapper");
}

fn wrapper_build(wrapper_dir: &Path, llama_src: &Path, target_os: &TargetOs) -> cc::Build {
    let mut build = cc::Build::new();

    build.cpp(true).warnings(false);

    for include_dir in wrapper_include_dirs(wrapper_dir, llama_src) {
        build.include(include_dir);
    }

    build.flag_if_supported("-std=c++17").pic(true);

    for source in wrapper_sources(wrapper_dir) {
        build.file(&source);
    }

    for flag in msvc_flags(target_os) {
        build.flag(flag);
    }

    if links_stdlib_statically(target_os) {
        build.cpp_link_stdlib(None);
    }

    build
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::cc_test_environment::with_cc_environment;
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

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
    #[serial]
    fn wrappers_are_compiled_into_an_archive() {
        let scratch = ScratchDir::new("cppwrapper-compile");
        std::fs::write(
            scratch.path().join("wrapper_probe.cpp"),
            b"extern \"C\" int llama_rs_probe() { return 7; }\n",
        )
        .expect("source must be writable");

        with_cc_environment(scratch.path(), || {
            compile_cpp_wrappers(
                scratch.path(),
                &scratch.path().join("llama.cpp"),
                &TargetOs::from_target_triple("aarch64-apple-darwin").expect("supported triple"),
            );
        });

        assert!(
            scratch
                .path()
                .join("libllama_cpp_bindings_sys_common_wrapper.a")
                .exists(),
            "the wrapper archive must be produced"
        );
    }

    #[test]
    fn platform_flag_decisions_cover_every_target() {
        let msvc = TargetOs::from_target_triple("x86_64-pc-windows-msvc").expect("msvc");
        let android = TargetOs::from_target_triple("aarch64-linux-android").expect("android");
        let apple = TargetOs::from_target_triple("aarch64-apple-darwin").expect("apple");

        assert_eq!(super::msvc_flags(&msvc), &["/std:c++17", "/EHsc"]);
        assert!(super::msvc_flags(&apple).is_empty());
        assert!(super::msvc_flags(&android).is_empty());

        assert_eq!(
            super::links_stdlib_statically(&android),
            cfg!(feature = "static-stdcxx")
        );
        assert!(!super::links_stdlib_statically(&apple));
    }

    #[test]
    fn the_builder_applies_flags_for_every_target_without_compiling() {
        let scratch = ScratchDir::new("cppwrapper-builder");
        std::fs::write(scratch.path().join("wrapper_a.cpp"), b"").expect("source must be writable");

        for triple in [
            "x86_64-pc-windows-msvc",
            "aarch64-linux-android",
            "aarch64-apple-darwin",
        ] {
            let target_os = TargetOs::from_target_triple(triple).expect(triple);

            let _ = super::wrapper_build(
                scratch.path(),
                &scratch.path().join("llama.cpp"),
                &target_os,
            );
        }
    }
}
