use std::path::Path;

use crate::BuildContext;
use crate::target_os::TargetOs;

pub fn test_build_context(root: &Path, llama_src: &Path, target_triple: &str) -> BuildContext {
    let target_os =
        TargetOs::from_target_triple(target_triple).expect("the triple must name a supported OS");

    BuildContext {
        manifest_dir: root.to_path_buf(),
        out_dir: root.to_path_buf(),
        target_dir: root.join("target"),
        cmake_dir: root.join("cmake-out"),
        llama_src: llama_src.to_path_buf(),
        target_os,
        target_triple: target_triple.to_owned(),
        host: target_triple.to_owned(),
        build_shared_libs: false,
        profile: "Release".to_owned(),
        static_crt: false,
        opt_level: "0".to_owned(),
        debug: false,
        android_ndk: None,
    }
}

#[cfg(test)]
mod tests {
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

    use super::test_build_context;

    #[test]
    fn every_path_is_rooted_in_the_scratch_dir_and_the_triple_is_parsed() {
        let scratch = ScratchDir::new("testcontext");
        let llama_src = scratch.path().join("llama.cpp");

        let context = test_build_context(scratch.path(), &llama_src, "aarch64-linux-android");

        assert!(matches!(context.target_os, TargetOs::Android));
        assert_eq!(context.target_triple, context.host);
        assert_eq!(context.manifest_dir, scratch.path());
        assert_eq!(context.llama_src, llama_src);
        assert!(context.cmake_dir.starts_with(scratch.path()));
        assert!(!context.static_crt);
    }
}
