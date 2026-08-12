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

const MTMD_SKIP_FILES: &[&str] = &["mtmd-cli.cpp", "deprecation-warning.cpp"];

fn mtmd_include_dirs(llama_src: &Path, mtmd_src: &Path) -> Vec<PathBuf> {
    vec![
        mtmd_src.to_path_buf(),
        llama_src.to_path_buf(),
        llama_src.join("include"),
        llama_src.join("ggml/include"),
        llama_src.join("common"),
        llama_src.join("vendor"),
    ]
}

fn mtmd_sources(mtmd_src: &Path) -> Vec<PathBuf> {
    let pattern = mtmd_src.join("**/*.cpp");

    let paths = match glob_paths::collect_paths(&pattern.to_string_lossy()) {
        Ok(paths) => paths,
        Err(error) => panic!("mtmd source discovery failed: {error}"),
    };

    paths
        .into_iter()
        .filter(|path| {
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();

            !MTMD_SKIP_FILES.contains(&filename)
        })
        .collect()
}

pub fn compile_mtmd(llama_src: &Path, target_os: &TargetOs) {
    mtmd_build(llama_src, target_os).compile("mtmd");
}

fn mtmd_build(llama_src: &Path, target_os: &TargetOs) -> cc::Build {
    let mtmd_src = llama_src.join("tools/mtmd");
    let mut build = cc::Build::new();

    build.cpp(true).warnings(false);

    for include_dir in mtmd_include_dirs(llama_src, &mtmd_src) {
        build.include(include_dir);
    }

    build.flag_if_supported("-std=c++17").pic(true);

    for flag in msvc_flags(target_os) {
        build.flag(flag);
    }

    if links_stdlib_statically(target_os) {
        build.cpp_link_stdlib(None);
    }

    for source in mtmd_sources(&mtmd_src) {
        build.file(&source);
    }

    build
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::cc_test_environment::with_cc_environment;
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

    use super::compile_mtmd;
    use super::mtmd_include_dirs;
    use super::mtmd_sources;

    fn llama_src_with_mtmd(scratch: &ScratchDir) -> std::path::PathBuf {
        let llama_src = scratch.path().join("llama.cpp");
        let mtmd_src = llama_src.join("tools/mtmd");
        std::fs::create_dir_all(&mtmd_src).expect("mtmd dir must be creatable");
        std::fs::write(
            mtmd_src.join("probe.cpp"),
            b"extern \"C\" int mtmd_probe() { return 3; }\n",
        )
        .expect("source must be writable");

        llama_src
    }

    #[test]
    fn the_include_list_covers_mtmd_and_upstream_trees() {
        let dirs = mtmd_include_dirs(
            std::path::Path::new("/llama"),
            std::path::Path::new("/llama/tools/mtmd"),
        );

        assert!(dirs.contains(&std::path::PathBuf::from("/llama/tools/mtmd")));
        assert!(dirs.contains(&std::path::PathBuf::from("/llama/vendor")));
    }

    #[test]
    fn the_skip_list_is_excluded_from_discovery() {
        let scratch = ScratchDir::new("mtmd-skip");
        let mtmd_src = scratch.path().join("tools/mtmd");
        std::fs::create_dir_all(mtmd_src.join("nested")).expect("dirs must be creatable");
        for name in ["kept.cpp", "mtmd-cli.cpp", "deprecation-warning.cpp"] {
            std::fs::write(mtmd_src.join(name), b"").expect("source must be writable");
        }
        std::fs::write(mtmd_src.join("nested/also-kept.cpp"), b"")
            .expect("nested source must be writable");

        let sources = mtmd_sources(&mtmd_src);

        assert_eq!(sources.len(), 2, "got: {sources:?}");
        assert!(sources.iter().any(|path| path.ends_with("kept.cpp")));
        assert!(sources.iter().any(|path| path.ends_with("also-kept.cpp")));
    }

    #[test]
    #[should_panic(expected = "mtmd source discovery failed")]
    fn a_tree_without_sources_panics() {
        let scratch = ScratchDir::new("mtmd-empty");

        let _ = mtmd_sources(scratch.path());
    }

    #[test]
    #[serial]
    fn mtmd_sources_are_compiled_into_an_archive() {
        let scratch = ScratchDir::new("mtmd-compile");
        let llama_src = llama_src_with_mtmd(&scratch);

        with_cc_environment(scratch.path(), || {
            compile_mtmd(
                &llama_src,
                &TargetOs::from_target_triple("aarch64-apple-darwin").expect("supported triple"),
            );
        });

        assert!(
            scratch.path().join("libmtmd.a").exists(),
            "the mtmd archive must be produced"
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
        let scratch = ScratchDir::new("mtmd-builder");
        let llama_src = llama_src_with_mtmd(&scratch);

        for triple in [
            "x86_64-pc-windows-msvc",
            "aarch64-linux-android",
            "aarch64-apple-darwin",
        ] {
            let target_os = TargetOs::from_target_triple(triple).expect(triple);

            let _ = super::mtmd_build(&llama_src, &target_os);
        }
    }
}
