use std::env;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

const CMAKE_AFFECTING_FEATURES: &[(&str, bool)] = &[
    ("cuda", cfg!(feature = "cuda")),
    ("cuda-no-vmm", cfg!(feature = "cuda-no-vmm")),
    ("metal", cfg!(feature = "metal")),
    ("vulkan", cfg!(feature = "vulkan")),
    ("rocm", cfg!(feature = "rocm")),
    ("openmp", cfg!(feature = "openmp")),
    ("dynamic-link", cfg!(feature = "dynamic-link")),
    ("dynamic-backends", cfg!(feature = "dynamic-backends")),
    ("system-ggml", cfg!(feature = "system-ggml")),
    ("system-ggml-static", cfg!(feature = "system-ggml-static")),
    ("shared-stdcxx", cfg!(feature = "shared-stdcxx")),
    ("static-stdcxx", cfg!(feature = "static-stdcxx")),
];

pub fn stable_cmake_build_dir(
    target_dir: &Path,
    target_triple: &str,
    profile: &str,
    static_crt: bool,
    build_shared_libs: bool,
) -> PathBuf {
    if let Ok(override_path) = env::var("LLAMA_CMAKE_BUILD_DIR_OVERRIDE") {
        let path = PathBuf::from(override_path);
        std::fs::create_dir_all(&path).expect("failed to create cmake build directory override");

        return path;
    }

    let mut hasher = DefaultHasher::new();
    target_triple.hash(&mut hasher);
    profile.hash(&mut hasher);
    static_crt.hash(&mut hasher);
    build_shared_libs.hash(&mut hasher);

    for (name, enabled) in CMAKE_AFFECTING_FEATURES {
        name.hash(&mut hasher);
        enabled.hash(&mut hasher);
    }

    let digest = format!("{:016x}", hasher.finish());
    let path = target_dir.join("llama-cpp-cmake-build").join(digest);

    std::fs::create_dir_all(&path).expect("failed to create cmake build directory");

    path
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::scratch_dir::ScratchDir;

    use super::stable_cmake_build_dir;

    const OVERRIDE_VAR: &str = "LLAMA_CMAKE_BUILD_DIR_OVERRIDE";

    fn build_dir_for(
        scratch: &ScratchDir,
        target_triple: &str,
        profile: &str,
    ) -> std::path::PathBuf {
        stable_cmake_build_dir(scratch.path(), target_triple, profile, false, false)
    }

    #[test]
    #[serial]
    fn the_same_inputs_always_yield_the_same_directory() {
        unsafe { std::env::remove_var(OVERRIDE_VAR) };
        let scratch = ScratchDir::new("cmakedir-stable");

        let first = build_dir_for(&scratch, "aarch64-apple-darwin", "Release");
        let second = build_dir_for(&scratch, "aarch64-apple-darwin", "Release");

        assert_eq!(first, second, "the directory must be stable across calls");
        assert!(first.is_dir(), "the directory must be created");
    }

    #[test]
    #[serial]
    fn each_configuration_input_changes_the_directory() {
        unsafe { std::env::remove_var(OVERRIDE_VAR) };
        let scratch = ScratchDir::new("cmakedir-varies");
        let baseline = build_dir_for(&scratch, "aarch64-apple-darwin", "Release");

        let other_triple = build_dir_for(&scratch, "x86_64-unknown-linux-gnu", "Release");
        let other_profile = build_dir_for(&scratch, "aarch64-apple-darwin", "Debug");
        let static_crt = stable_cmake_build_dir(
            scratch.path(),
            "aarch64-apple-darwin",
            "Release",
            true,
            false,
        );
        let shared_libs = stable_cmake_build_dir(
            scratch.path(),
            "aarch64-apple-darwin",
            "Release",
            false,
            true,
        );

        for (label, candidate) in [
            ("target triple", other_triple),
            ("profile", other_profile),
            ("static crt", static_crt),
            ("shared libs", shared_libs),
        ] {
            assert_ne!(baseline, candidate, "{label} must change the build dir");
        }
    }

    #[test]
    #[serial]
    fn the_override_environment_variable_wins_and_is_created() {
        let scratch = ScratchDir::new("cmakedir-override");
        let override_path = scratch.path().join("explicit-build-dir");
        unsafe { std::env::set_var(OVERRIDE_VAR, &override_path) };

        let resolved = build_dir_for(&scratch, "aarch64-apple-darwin", "Release");

        unsafe { std::env::remove_var(OVERRIDE_VAR) };

        assert_eq!(resolved, override_path);
        assert!(resolved.is_dir(), "the override directory must be created");
    }
}
