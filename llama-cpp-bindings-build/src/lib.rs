mod android_ndk;
mod bindgen_config;
#[cfg(test)]
mod cc_test_environment;
mod cmake_config;
mod cpp_wrapper;
mod cpp_wrapper_mtmd;
mod glob_paths;
mod host_platform;
mod library_asset_extraction;
mod library_linking;
mod library_name_extraction;
mod rebuild_tracking;
#[cfg(test)]
mod scratch_dir;
mod shared_libs;
mod stable_cmake_build_dir;
mod target_os;

use std::env;
use std::path::{Path, PathBuf};

use android_ndk::AndroidNdk;
use stable_cmake_build_dir::stable_cmake_build_dir;
use target_os::TargetOs;

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

#[derive(Debug)]
pub struct BuildContext {
    pub manifest_dir: PathBuf,
    pub out_dir: PathBuf,
    pub target_dir: PathBuf,
    pub cmake_dir: PathBuf,
    pub llama_src: PathBuf,
    pub target_os: TargetOs,
    pub target_triple: String,
    pub build_shared_libs: bool,
    pub profile: String,
    pub static_crt: bool,
    pub android_ndk: Option<AndroidNdk>,
}

impl BuildContext {
    fn detect() -> Self {
        let target_triple =
            env::var("TARGET").expect("TARGET env var is required in build scripts");
        let target_os = TargetOs::from_target_triple(&target_triple)
            .unwrap_or_else(|error| panic!("Failed to parse target OS: {error}"));
        let out_dir = PathBuf::from(
            env::var("OUT_DIR").expect("OUT_DIR env var is required in build scripts"),
        );
        let target_dir = cargo_target_dir(&out_dir);
        let manifest_dir = env::var("CARGO_MANIFEST_DIR")
            .expect("CARGO_MANIFEST_DIR env var is required in build scripts");
        let manifest_dir = PathBuf::from(manifest_dir);
        let llama_src = manifest_dir.join("llama.cpp");

        let build_shared_libs = env::var("LLAMA_BUILD_SHARED_LIBS")
            .map_or_else(|_| cfg!(feature = "dynamic-link"), |value| value == "1");

        let profile = env::var("LLAMA_LIB_PROFILE").unwrap_or_else(|_| "Release".to_string());

        let static_crt = env::var("LLAMA_STATIC_CRT")
            .map(|value| value == "1")
            .unwrap_or(false);

        let android_ndk = if target_os.is_android() {
            Some(
                AndroidNdk::detect(&target_triple)
                    .unwrap_or_else(|error| panic!("Android NDK detection failed: {error}")),
            )
        } else {
            None
        };

        let cmake_dir = stable_cmake_build_dir(
            &target_dir,
            &target_triple,
            &profile,
            static_crt,
            build_shared_libs,
        );

        debug_log!("TARGET: {}", target_triple);
        debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir.display());
        debug_log!("TARGET_DIR: {}", target_dir.display());
        debug_log!("OUT_DIR: {}", out_dir.display());
        debug_log!("CMAKE_DIR: {}", cmake_dir.display());
        debug_log!("BUILD_SHARED: {}", build_shared_libs);

        Self {
            manifest_dir,
            out_dir,
            target_dir,
            cmake_dir,
            llama_src,
            target_os,
            target_triple,
            build_shared_libs,
            profile,
            static_crt,
            android_ndk,
        }
    }
}

fn cargo_target_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR is not deep enough to determine target directory")
        .to_path_buf()
}

pub fn build() {
    let context = BuildContext::detect();

    rebuild_tracking::register_rebuild_triggers(&context.manifest_dir, &context.llama_src);

    bindgen_config::generate_bindings(
        &context.manifest_dir,
        &context.llama_src,
        &context.out_dir,
        &context.target_os,
        &context.target_triple,
        context.android_ndk.as_ref(),
    );

    cpp_wrapper::compile_cpp_wrappers(
        &context.manifest_dir,
        &context.llama_src,
        &context.target_os,
    );

    let build_dir = cmake_config::configure_and_build(&context);

    cpp_wrapper_mtmd::compile_mtmd(&context.llama_src, &context.target_os);

    library_linking::link_libraries(
        &context.cmake_dir,
        &build_dir,
        &context.target_os,
        &context.target_triple,
        context.build_shared_libs,
        &context.profile,
    );

    if context.build_shared_libs {
        shared_libs::copy_shared_libraries(&context.cmake_dir, &context.target_dir);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serial_test::serial;

    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

    use super::BuildContext;
    use super::cargo_target_dir;

    /// A complete stand-in for the sys crate: wrapper headers and sources, and
    /// a minimal CMake project in place of llama.cpp. Lets `build()` run end to
    /// end in about a second instead of compiling upstream.
    fn synthetic_sys_crate(scratch: &ScratchDir) -> std::path::PathBuf {
        let manifest_dir = scratch.path().join("sys");
        let llama_src = manifest_dir.join("llama.cpp");

        for relative in ["include", "ggml/include", "common", "vendor", "tools/mtmd"] {
            std::fs::create_dir_all(llama_src.join(relative)).expect("tree must be creatable");
        }
        std::fs::create_dir_all(manifest_dir.join("GSL/include")).expect("gsl must be creatable");

        std::fs::write(manifest_dir.join("wrapper.h"), b"int llama_probe(void);\n")
            .expect("header must be writable");
        std::fs::write(
            manifest_dir.join("wrapper_mtmd.h"),
            b"int mtmd_probe(void);\n",
        )
        .expect("header must be writable");
        std::fs::write(
            manifest_dir.join("wrapper_probe.cpp"),
            b"extern \"C\" int llama_probe() { return 1; }\n",
        )
        .expect("wrapper source must be writable");
        std::fs::write(
            llama_src.join("tools/mtmd/probe.cpp"),
            b"extern \"C\" int mtmd_probe() { return 2; }\n",
        )
        .expect("mtmd source must be writable");
        std::fs::write(
            llama_src.join("CMakeLists.txt"),
            b"cmake_minimum_required(VERSION 3.14)\n\
              project(probe C)\n\
              add_library(probe probe.c)\n\
              install(TARGETS probe ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)\n",
        )
        .expect("CMakeLists must be writable");
        std::fs::write(
            llama_src.join("probe.c"),
            b"int probe(void) { return 1; }\n",
        )
        .expect("cmake source must be writable");

        manifest_dir
    }

    fn run_build_against(manifest_dir: &Path, shared_libs: &str) {
        let out_dir = manifest_dir.join("target/debug/build/sys-abc/out");
        std::fs::create_dir_all(&out_dir).expect("out dir must be creatable");
        let rocm_root = manifest_dir.join("rocm");
        std::fs::create_dir_all(rocm_root.join("lib")).expect("rocm lib must be creatable");
        let host = format!("{}-apple-darwin", std::env::consts::ARCH);

        unsafe {
            std::env::set_var("ROCM_PATH", &rocm_root);
            std::env::set_var("VULKAN_SDK", manifest_dir);
            std::env::set_var("CARGO_MANIFEST_DIR", manifest_dir);
            std::env::set_var("OUT_DIR", &out_dir);
            std::env::set_var("TARGET", &host);
            std::env::set_var("HOST", &host);
            std::env::set_var("OPT_LEVEL", "0");
            std::env::set_var("PROFILE", "debug");
            std::env::set_var("NUM_JOBS", "1");
            std::env::set_var("LLAMA_BUILD_SHARED_LIBS", shared_libs);
            std::env::remove_var("LLAMA_CMAKE_BUILD_DIR_OVERRIDE");
            std::env::remove_var("LLAMA_LIB_PROFILE");
            std::env::remove_var("LLAMA_STATIC_CRT");
        }

        super::build();

        unsafe {
            std::env::remove_var("LLAMA_BUILD_SHARED_LIBS");
            std::env::remove_var("OPT_LEVEL");
            std::env::remove_var("PROFILE");
            std::env::remove_var("NUM_JOBS");
            std::env::remove_var("ROCM_PATH");
            std::env::remove_var("VULKAN_SDK");
            std::env::remove_var("TrackFileAccess");
        }
    }

    #[test]
    #[serial]
    fn a_static_build_runs_every_stage() {
        let scratch = ScratchDir::new("build-static");

        run_build_against(&synthetic_sys_crate(&scratch), "0");
    }

    #[test]
    #[serial]
    fn a_shared_build_also_copies_the_libraries() {
        let scratch = ScratchDir::new("build-shared");

        run_build_against(&synthetic_sys_crate(&scratch), "1");
    }

    #[test]
    fn the_target_dir_is_three_levels_above_out_dir() {
        let out_dir = Path::new("/w/target/debug/build/llama-cpp-bindings-sys-abc123/out");

        assert_eq!(cargo_target_dir(out_dir), Path::new("/w/target/debug"));
    }

    #[test]
    #[should_panic(expected = "OUT_DIR is not deep enough")]
    fn a_shallow_out_dir_panics() {
        let _ = cargo_target_dir(Path::new("/out"));
    }

    #[test]
    #[serial]
    fn detect_reads_the_build_script_environment() {
        let scratch = ScratchDir::new("context-detect");
        let out_dir = scratch.path().join("target/debug/build/crate-abc/out");
        std::fs::create_dir_all(&out_dir).expect("out dir must be creatable");

        unsafe {
            std::env::set_var("TARGET", "aarch64-apple-darwin");
            std::env::set_var("OUT_DIR", &out_dir);
            std::env::set_var("CARGO_MANIFEST_DIR", scratch.path());
            std::env::remove_var("LLAMA_CMAKE_BUILD_DIR_OVERRIDE");
            std::env::remove_var("LLAMA_BUILD_SHARED_LIBS");
            std::env::remove_var("LLAMA_LIB_PROFILE");
            std::env::remove_var("LLAMA_STATIC_CRT");
        }

        let context = BuildContext::detect();

        assert!(matches!(context.target_os, TargetOs::Apple(_)));
        assert_eq!(context.target_triple, "aarch64-apple-darwin");
        assert_eq!(context.manifest_dir, scratch.path());
        assert_eq!(context.llama_src, scratch.path().join("llama.cpp"));
        assert_eq!(context.target_dir, scratch.path().join("target/debug"));
        assert_eq!(context.profile, "Release");
        assert!(!context.static_crt);
        assert_eq!(
            context.build_shared_libs,
            cfg!(feature = "dynamic-link"),
            "with no override, shared linking follows the dynamic-link feature"
        );
        assert!(context.cmake_dir.is_dir());
        assert!(context.android_ndk.is_none());
        assert!(format!("{context:?}").contains("BuildContext"));
    }

    #[test]
    #[serial]
    fn detect_honours_the_profile_and_crt_overrides() {
        let scratch = ScratchDir::new("context-overrides");
        let out_dir = scratch.path().join("target/debug/build/crate-abc/out");
        std::fs::create_dir_all(&out_dir).expect("out dir must be creatable");

        unsafe {
            std::env::set_var("TARGET", "x86_64-unknown-linux-gnu");
            std::env::set_var("OUT_DIR", &out_dir);
            std::env::set_var("CARGO_MANIFEST_DIR", scratch.path());
            std::env::set_var("LLAMA_LIB_PROFILE", "Debug");
            std::env::set_var("LLAMA_STATIC_CRT", "1");
            std::env::set_var("LLAMA_BUILD_SHARED_LIBS", "1");
        }

        let context = BuildContext::detect();

        unsafe {
            std::env::remove_var("LLAMA_LIB_PROFILE");
            std::env::remove_var("LLAMA_STATIC_CRT");
            std::env::remove_var("LLAMA_BUILD_SHARED_LIBS");
        }

        assert_eq!(context.profile, "Debug");
        assert!(context.static_crt);
        assert!(context.build_shared_libs);
        assert!(matches!(context.target_os, TargetOs::Linux));
    }

    #[test]
    #[serial]
    #[should_panic(expected = "Android NDK detection failed")]
    fn an_android_target_without_an_ndk_panics() {
        let scratch = ScratchDir::new("context-android");
        let out_dir = scratch.path().join("target/debug/build/crate-abc/out");
        std::fs::create_dir_all(&out_dir).expect("out dir must be creatable");

        unsafe {
            std::env::set_var("TARGET", "aarch64-linux-android");
            std::env::set_var("OUT_DIR", &out_dir);
            std::env::set_var("CARGO_MANIFEST_DIR", scratch.path());
            for name in [
                "ANDROID_NDK",
                "ANDROID_NDK_ROOT",
                "NDK_ROOT",
                "CARGO_NDK_ANDROID_NDK",
                "ANDROID_SDK_ROOT",
            ] {
                std::env::remove_var(name);
            }
            std::env::set_var("ANDROID_HOME", scratch.path());
        }

        let _ = BuildContext::detect();
    }
}
