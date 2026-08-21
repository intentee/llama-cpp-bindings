use std::path::Path;
use std::path::PathBuf;

use crate::BuildContext;
use crate::cpp_build::cpp_build;
use crate::glob_paths;

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
    let paths = match glob_paths::collect_paths(mtmd_src, "**/*.cpp") {
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

pub fn compile_mtmd(context: &BuildContext) {
    mtmd_build(context).compile("mtmd");
}

fn mtmd_build(context: &BuildContext) -> cc::Build {
    let mtmd_src = context.llama_src.join("tools/mtmd");

    cpp_build(
        context,
        mtmd_include_dirs(&context.llama_src, &mtmd_src),
        mtmd_sources(&mtmd_src),
    )
}

#[cfg(test)]
mod tests {
    use crate::host_platform::HostPlatform;
    use crate::host_target_triple::host_target_triple;
    use crate::scratch_dir::ScratchDir;
    use crate::test_build_context::test_build_context;

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
    fn mtmd_sources_are_compiled_into_an_archive() {
        let scratch = ScratchDir::new("mtmd-compile");
        let llama_src = llama_src_with_mtmd(&scratch);
        let context = test_build_context(scratch.path(), &llama_src, &host_target_triple());

        compile_mtmd(&context);

        let archive = HostPlatform::current().static_library_file_name("mtmd");

        assert!(
            scratch.path().join(&archive).exists(),
            "the mtmd archive {archive} must be produced"
        );
    }
}
