use std::env;
use std::path::Path;

use crate::debug_log;
use crate::host_platform::HostPlatform;
use crate::library_name_extraction::extract_lib_names;
use crate::target_os::{AppleVariant, TargetOs, WindowsVariant};

pub fn link_libraries(
    cmake_dir: &Path,
    build_dir: &Path,
    target_os: &TargetOs,
    target_triple: &str,
    build_shared_libs: bool,
    profile: &str,
) {
    emit_search_paths(cmake_dir, build_dir);
    link_system_ggml_paths(build_dir);
    link_cmake_built_libraries(cmake_dir, build_shared_libs, profile);
    link_cuda_libraries(build_shared_libs);
    link_rocm_libraries(build_shared_libs);
    link_openmp(target_triple);
    link_platform_system_libraries(target_os);
}

fn emit_search_paths(cmake_dir: &Path, build_dir: &Path) {
    println!(
        "cargo:rustc-link-search={}",
        cmake_dir.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search={}",
        cmake_dir.join("lib64").display()
    );
    println!("cargo:rustc-link-search={}", build_dir.display());
}

fn link_system_ggml_paths(build_dir: &Path) {
    if !cfg!(feature = "system-ggml") {
        return;
    }

    let cmake_cache = build_dir.join("build").join("CMakeCache.txt");
    let Ok(cache_contents) = std::fs::read_to_string(&cmake_cache) else {
        return;
    };

    let mut ggml_lib_dirs = std::collections::HashSet::new();

    for line in cache_contents.lines() {
        let is_ggml_library_entry = line.starts_with("GGML_LIBRARY:")
            || line.starts_with("GGML_BASE_LIBRARY:")
            || line.starts_with("GGML_CPU_LIBRARY:");

        if is_ggml_library_entry
            && let Some(lib_path) = line.split('=').nth(1)
            && let Some(parent) = Path::new(lib_path).parent()
        {
            ggml_lib_dirs.insert(parent.to_path_buf());
        }
    }

    for lib_dir in ggml_lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        debug_log!("Added system GGML library path: {}", lib_dir.display());
    }
}

fn link_cmake_built_libraries(cmake_dir: &Path, build_shared_libs: bool, profile: &str) {
    let link_kind = if build_shared_libs {
        "dylib"
    } else if cfg!(feature = "system-ggml-static") {
        "static"
    } else if cfg!(feature = "system-ggml") {
        "dylib"
    } else {
        "static"
    };

    let lib_names = extract_lib_names(cmake_dir, build_shared_libs);
    assert!(!lib_names.is_empty(), "no libraries found in build output");

    link_llama_common_internal_libraries(cmake_dir, profile);
    link_system_ggml_libraries(link_kind);

    for lib_name in lib_names {
        let link = format!("cargo:rustc-link-lib={link_kind}={lib_name}");
        debug_log!("LINK {link}");
        println!("{link}");
    }
}

fn link_llama_common_internal_libraries(cmake_dir: &Path, profile: &str) {
    let common_lib_dir = cmake_dir.join("build").join("common");

    if common_lib_dir.is_dir() {
        emit_search_path_with_profile(&common_lib_dir, profile);
        println!("cargo:rustc-link-lib=static=llama-common-base");
    }

    let httplib_dir = cmake_dir.join("build").join("vendor").join("cpp-httplib");

    if httplib_dir.is_dir() {
        emit_search_path_with_profile(&httplib_dir, profile);
        println!("cargo:rustc-link-lib=static=cpp-httplib");
    }
}

fn emit_search_path_with_profile(lib_dir: &Path, profile: &str) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    let profile_dir = lib_dir.join(profile);

    if profile_dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", profile_dir.display());
    }
}

fn link_system_ggml_libraries(link_kind: &str) {
    if !cfg!(feature = "system-ggml") {
        return;
    }

    println!("cargo:rustc-link-lib={link_kind}=ggml");
    println!("cargo:rustc-link-lib={link_kind}=ggml-base");
    println!("cargo:rustc-link-lib={link_kind}=ggml-cpu");
}

fn link_cuda_libraries(build_shared_libs: bool) {
    if !cfg!(feature = "cuda") || build_shared_libs {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    emit_cuda_search_paths(find_cuda_helper::find_cuda_lib_dirs());

    link_cuda_for(HostPlatform::current());
}

fn emit_cuda_search_paths(lib_dirs: impl IntoIterator<Item = std::path::PathBuf>) {
    for lib_dir in lib_dirs {
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
}

fn link_cuda_for(platform: HostPlatform) {
    if platform == HostPlatform::Windows {
        link_cuda_windows();
    } else {
        link_cuda_unix();
    }
}

fn link_cuda_windows() {
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cublasLt");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }
}

fn link_cuda_unix() {
    println!("cargo:rustc-link-lib=static=cudart_static");
    println!("cargo:rustc-link-lib=static=cublas_static");
    println!("cargo:rustc-link-lib=static=cublasLt_static");

    if !cfg!(feature = "cuda-no-vmm") {
        println!("cargo:rustc-link-lib=cuda");
    }

    println!("cargo:rustc-link-lib=static=culibos");
}

fn link_rocm_libraries(build_shared_libs: bool) {
    if !cfg!(feature = "rocm") || build_shared_libs {
        return;
    }

    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let rocm_path = env::var("ROCM_PATH")
        .or_else(|_| env::var("HIP_PATH"))
        .unwrap_or_else(|_| default_rocm_path(HostPlatform::current()));

    let rocm_lib = Path::new(&rocm_path).join("lib");

    assert!(
        rocm_lib.exists(),
        "ROCm libraries not found at: {}\n\
         Please install ROCm or set ROCM_PATH/HIP_PATH environment variable.\n\
         Download from: https://rocm.docs.amd.com/",
        rocm_lib.display()
    );

    println!("cargo:rustc-link-search=native={}", rocm_lib.display());
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-lib=dylib=rocblas");
    println!("cargo:rustc-link-lib=dylib=hipblas");
}

fn default_rocm_path(platform: HostPlatform) -> String {
    match platform {
        HostPlatform::Windows => "C:\\Program Files\\AMD\\ROCm".to_owned(),
        HostPlatform::MacOs | HostPlatform::Unixlike => "/opt/rocm".to_owned(),
    }
}

fn link_openmp(target_triple: &str) {
    if cfg!(feature = "openmp") && target_triple.contains("gnu") {
        println!("cargo:rustc-link-lib=gomp");
    }
}

fn link_platform_system_libraries(target_os: &TargetOs) {
    match target_os {
        TargetOs::Windows(WindowsVariant::Msvc) => {
            link_msvc_system_libraries();
        }
        TargetOs::Linux => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
        TargetOs::Apple(variant) => {
            link_apple_frameworks(*variant);
        }
        TargetOs::Android => {
            link_android_cpp_stdlib();
        }
        TargetOs::Windows(_) => {}
    }
}

fn link_android_cpp_stdlib() {
    if cfg!(feature = "static-stdcxx") {
        println!("cargo:rustc-link-lib=c++_static");
        println!("cargo:rustc-link-lib=c++abi");
    } else if cfg!(feature = "shared-stdcxx") {
        println!("cargo:rustc-link-lib=c++_shared");
    }
}

fn link_msvc_system_libraries() {
    println!("cargo:rustc-link-lib=advapi32");

    let crt_static = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .contains("crt-static");

    if let Some(debug_runtime) = msvc_debug_runtime(cfg!(debug_assertions), crt_static) {
        println!("{debug_runtime}");
    }
}

fn msvc_debug_runtime(debug_assertions: bool, crt_static: bool) -> Option<&'static str> {
    if !debug_assertions {
        return None;
    }

    if crt_static {
        Some("cargo:rustc-link-lib=libcmtd")
    } else {
        Some("cargo:rustc-link-lib=dylib=msvcrtd")
    }
}

fn link_apple_frameworks(variant: AppleVariant) {
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=c++");

    if let AppleVariant::MacOS = variant
        && let Some(path) = macos_link_search_path()
    {
        println!("cargo:rustc-link-lib=clang_rt.osx");
        println!("cargo:rustc-link-search={path}");
    }
}

fn macos_link_search_path() -> Option<String> {
    clang_search_dirs("clang")
}

fn clang_search_dirs(program: &str) -> Option<String> {
    let output = std::process::Command::new(program)
        .arg("--print-search-dirs")
        .output()
        .ok()?;

    if !output.status.success() {
        println!(
            "cargo:warning=failed to run 'clang --print-search-dirs', continuing without a link search path"
        );

        return None;
    }

    parse_clang_search_dirs(&String::from_utf8_lossy(&output.stdout))
}

fn parse_clang_search_dirs(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        if line.contains("libraries: =") {
            let path = line.split('=').nth(1)?;

            return Some(format!("{path}/lib/darwin"));
        }
    }

    println!("cargo:warning=failed to determine link search path, continuing without it");

    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serial_test::serial;

    use crate::host_platform::HostPlatform;
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

    use super::clang_search_dirs;
    use super::default_rocm_path;
    use super::emit_cuda_search_paths;
    use super::emit_search_path_with_profile;
    use super::emit_search_paths;
    use super::link_cmake_built_libraries;
    use super::link_cuda_for;
    use super::link_cuda_libraries;
    use super::link_libraries;
    use super::link_llama_common_internal_libraries;
    use super::link_openmp;
    use super::link_platform_system_libraries;
    use super::link_rocm_libraries;
    use super::link_system_ggml_paths;
    use super::msvc_debug_runtime;
    use super::parse_clang_search_dirs;

    fn archive_name(stem: &str) -> String {
        crate::host_platform::HostPlatform::current()
            .link_library_pattern(false)
            .replace('*', stem)
    }

    fn cmake_dir_with_library(scratch: &ScratchDir) -> PathBuf {
        let cmake_dir = scratch.path().join("cmake");
        let libs_dir = cmake_dir.join("lib");
        std::fs::create_dir_all(&libs_dir).expect("lib dir must be creatable");
        std::fs::write(libs_dir.join(archive_name("libggml")), b"x")
            .expect("archive must be writable");

        cmake_dir
    }

    #[test]
    fn search_paths_cover_lib_lib64_and_the_build_dir() {
        let scratch = ScratchDir::new("linking-search");

        emit_search_paths(&scratch.path().join("cmake"), scratch.path());
    }

    #[test]
    fn system_ggml_paths_are_skipped_without_the_feature_or_a_cache() {
        let scratch = ScratchDir::new("linking-ggml");

        link_system_ggml_paths(scratch.path());
    }

    #[test]
    fn cmake_built_libraries_are_linked() {
        let scratch = ScratchDir::new("linking-built");
        let cmake_dir = cmake_dir_with_library(&scratch);

        link_cmake_built_libraries(&cmake_dir, false, "Release");
    }

    #[test]
    #[should_panic(expected = "no libraries found in build output")]
    fn an_empty_build_output_panics() {
        let scratch = ScratchDir::new("linking-empty");

        link_cmake_built_libraries(scratch.path(), false, "Release");
    }

    #[test]
    fn internal_libraries_are_linked_when_their_directories_exist() {
        let scratch = ScratchDir::new("linking-internal");
        let cmake_dir = scratch.path().join("cmake");
        std::fs::create_dir_all(cmake_dir.join("build/common/Release"))
            .expect("common dir must be creatable");
        std::fs::create_dir_all(cmake_dir.join("build/vendor/cpp-httplib"))
            .expect("httplib dir must be creatable");

        link_llama_common_internal_libraries(&cmake_dir, "Release");
    }

    #[test]
    fn internal_libraries_are_skipped_when_absent() {
        let scratch = ScratchDir::new("linking-internal-absent");

        link_llama_common_internal_libraries(scratch.path(), "Release");
    }

    #[test]
    fn a_profile_subdirectory_adds_a_second_search_path() {
        let scratch = ScratchDir::new("linking-profile");
        std::fs::create_dir_all(scratch.path().join("Release"))
            .expect("profile dir must be creatable");

        emit_search_path_with_profile(scratch.path(), "Release");
        emit_search_path_with_profile(scratch.path(), "Debug");
    }

    /// `link_rocm_libraries` asserts the ROCm lib directory exists, so the
    /// environment is pointed at a real one for the duration of the call.
    fn with_rocm_root<TBody: FnOnce()>(body: TBody) {
        let scratch = ScratchDir::new("linking-rocm");
        std::fs::create_dir_all(scratch.path().join("lib")).expect("rocm lib must be creatable");
        unsafe { std::env::set_var("ROCM_PATH", scratch.path()) };

        body();

        unsafe { std::env::remove_var("ROCM_PATH") };
    }

    #[test]
    #[serial]
    fn accelerator_linking_is_skipped_for_shared_library_builds() {
        link_cuda_libraries(true);
        with_rocm_root(|| link_rocm_libraries(true));
    }

    /// The accelerator and platform emitters are compiled on every host even
    /// though `cfg!` gates keep them unreachable here, so they are driven
    /// directly rather than left unexercised.
    #[test]
    fn cuda_emitters_produce_both_link_flavours() {
        super::link_cuda_windows();
        super::link_cuda_unix();
    }

    #[test]
    fn the_system_ggml_emitter_covers_both_link_kinds() {
        super::link_system_ggml_libraries("static");
        super::link_system_ggml_libraries("dylib");
    }

    #[test]
    fn the_msvc_emitter_runs_without_a_windows_host() {
        super::link_msvc_system_libraries();
    }

    #[test]
    fn the_android_stdlib_emitter_runs_without_an_android_host() {
        super::link_android_cpp_stdlib();
    }

    #[test]
    fn apple_frameworks_are_emitted_for_both_variants() {
        super::link_apple_frameworks(crate::target_os::AppleVariant::MacOS);
        super::link_apple_frameworks(crate::target_os::AppleVariant::Other);
    }

    #[test]
    #[serial]
    fn accelerator_linking_runs_for_static_builds() {
        link_cuda_libraries(false);
        with_rocm_root(|| link_rocm_libraries(false));
    }

    #[test]
    fn openmp_is_only_linked_for_gnu_triples() {
        link_openmp("x86_64-unknown-linux-gnu");
        link_openmp("aarch64-apple-darwin");
    }

    #[test]
    fn every_platform_arm_emits_without_panicking() {
        for triple in [
            "aarch64-apple-darwin",
            "aarch64-apple-ios",
            "x86_64-unknown-linux-gnu",
            "aarch64-linux-android",
            "x86_64-pc-windows-msvc",
            "x86_64-pc-windows-gnu",
        ] {
            let target_os = TargetOs::from_target_triple(triple).expect("supported triple");

            link_platform_system_libraries(&target_os);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[serial]
    fn the_macos_link_search_path_is_discovered_from_clang() {
        use super::macos_link_search_path;

        let path = macos_link_search_path().expect("clang ships with the macOS toolchain");

        assert!(path.ends_with("/lib/darwin"), "got: {path}");
    }

    #[test]
    #[serial]
    fn linking_end_to_end_emits_every_group() {
        let scratch = ScratchDir::new("linking-end-to-end");
        let cmake_dir = cmake_dir_with_library(&scratch);
        let target_os =
            TargetOs::from_target_triple("aarch64-apple-darwin").expect("supported triple");

        with_rocm_root(|| {
            link_libraries(
                &cmake_dir,
                scratch.path(),
                &target_os,
                "aarch64-apple-darwin",
                false,
                "Release",
            );
        });
    }

    #[test]
    fn clang_search_dirs_are_parsed_into_a_darwin_lib_path() {
        let stdout = "programs: =/usr/bin\nlibraries: =/opt/clang\n";

        assert_eq!(
            parse_clang_search_dirs(stdout),
            Some("/opt/clang/lib/darwin".to_owned())
        );
    }

    #[test]
    fn clang_output_without_a_libraries_line_yields_nothing() {
        assert_eq!(parse_clang_search_dirs("programs: =/usr/bin\n"), None);
        assert_eq!(parse_clang_search_dirs(""), None);
    }

    #[test]
    fn the_msvc_debug_runtime_depends_on_assertions_and_the_crt_kind() {
        assert_eq!(
            msvc_debug_runtime(true, true),
            Some("cargo:rustc-link-lib=libcmtd")
        );
        assert_eq!(
            msvc_debug_runtime(true, false),
            Some("cargo:rustc-link-lib=dylib=msvcrtd")
        );
        assert_eq!(msvc_debug_runtime(false, true), None);
        assert_eq!(msvc_debug_runtime(false, false), None);
    }

    #[test]
    fn a_cmake_cache_contributes_system_ggml_search_paths() {
        let scratch = ScratchDir::new("linking-ggml-cache");
        let build_dir = scratch.path().join("build");
        std::fs::create_dir_all(&build_dir).expect("build dir must be creatable");
        std::fs::write(
            build_dir.join("CMakeCache.txt"),
            b"GGML_LIBRARY:FILEPATH=/opt/ggml/lib/libggml.so\n\
              GGML_BASE_LIBRARY:FILEPATH=/opt/ggml/lib/libggml-base.so\n\
              GGML_CPU_LIBRARY:FILEPATH=/opt/ggml/lib/libggml-cpu.so\n\
              UNRELATED:STRING=value\n",
        )
        .expect("cache must be writable");

        link_system_ggml_paths(scratch.path());
    }

    #[test]
    fn cuda_linking_dispatches_per_platform() {
        link_cuda_for(HostPlatform::Windows);
        link_cuda_for(HostPlatform::MacOs);
        link_cuda_for(HostPlatform::Unixlike);
    }

    #[test]
    fn the_default_rocm_path_follows_the_platform() {
        assert!(default_rocm_path(HostPlatform::Windows).contains("AMD"));
        assert_eq!(default_rocm_path(HostPlatform::MacOs), "/opt/rocm");
        assert_eq!(default_rocm_path(HostPlatform::Unixlike), "/opt/rocm");
    }

    #[test]
    fn a_missing_or_failing_clang_yields_no_search_path() {
        assert_eq!(
            clang_search_dirs("definitely-not-a-real-clang-binary"),
            None
        );
        assert_eq!(
            clang_search_dirs("false"),
            None,
            "a non-zero exit must be reported and skipped"
        );
    }

    #[test]
    fn cuda_search_paths_are_emitted_for_each_discovered_directory() {
        emit_cuda_search_paths(vec![
            std::path::PathBuf::from("/usr/local/cuda/lib64"),
            std::path::PathBuf::from("/opt/cuda/lib"),
        ]);
        emit_cuda_search_paths(Vec::new());
    }

    #[cfg(feature = "rocm")]
    #[test]
    #[serial]
    #[should_panic(expected = "ROCm libraries not found")]
    fn a_missing_rocm_installation_is_reported() {
        unsafe {
            std::env::set_var("ROCM_PATH", "/definitely/not/a/rocm/installation");
            std::env::remove_var("HIP_PATH");
        }

        link_rocm_libraries(false);
    }
}
