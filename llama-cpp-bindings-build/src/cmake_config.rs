use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;

use crate::BuildContext;
use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::target_os::{TargetOs, WindowsVariant};

pub fn configure_and_build(context: &BuildContext) -> PathBuf {
    let mut config = Config::new(&context.llama_src);

    configure_base_defines(&mut config);
    pass_cmake_env_vars(&mut config);
    configure_compiler_launchers(&mut config);
    configure_cpu_features(&mut config, &context.target_triple);
    configure_shared_libs(&mut config, context.build_shared_libs);
    configure_platform_specific(
        &mut config,
        &context.target_os,
        &context.target_triple,
        &context.profile,
        context.android_ndk.as_ref(),
    );
    configure_gpu_backends(&mut config, &context.target_os);
    configure_openmp(&mut config, &context.target_os);
    configure_system_ggml(&mut config);
    let backends_dir = configure_dynamic_backends(&mut config, &context.cmake_dir);

    config.static_crt(context.static_crt);
    config
        .out_dir(&context.cmake_dir)
        .profile(&context.profile)
        .very_verbose(env::var("CMAKE_VERBOSE").is_ok())
        .always_configure(false);

    let install_dir = config.build();

    if let Some(dir) = backends_dir {
        println!("cargo:backends_dir={}", dir.display());
    }

    install_dir
}

fn configure_dynamic_backends(config: &mut Config, cmake_dir: &Path) -> Option<PathBuf> {
    if !cfg!(feature = "dynamic-backends") {
        return None;
    }

    let backends_dir = cmake_dir.join("backends");

    std::fs::create_dir_all(&backends_dir).expect("failed to create backends directory");

    config.define("GGML_BACKEND_DL", "ON");
    config.define("GGML_CPU_ALL_VARIANTS", "ON");
    config.define(
        "GGML_BACKEND_DIR",
        backends_dir
            .to_str()
            .expect("backends directory must be valid UTF-8"),
    );

    Some(backends_dir)
}

fn configure_base_defines(config: &mut Config) {
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    config.define("LLAMA_BUILD_APP", "OFF");
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_CURL", "OFF");
    config.cflag("-w");
    config.cxxflag("-w");
}

fn configure_compiler_launchers(config: &mut Config) {
    println!("cargo:rerun-if-env-changed=LLAMA_DISABLE_CCACHE");

    if env::var("LLAMA_DISABLE_CCACHE").is_ok() {
        return;
    }

    let Some(ccache) = which("ccache") else {
        return;
    };

    let ccache_str = ccache.display().to_string();
    debug_log!("Using ccache for compilation: {ccache_str}");

    config.define("CMAKE_C_COMPILER_LAUNCHER", &ccache_str);
    config.define("CMAKE_CXX_COMPILER_LAUNCHER", &ccache_str);
    config.define("CMAKE_CUDA_COMPILER_LAUNCHER", &ccache_str);
}

fn which(program: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;

    for entry in env::split_paths(&path) {
        let candidate = entry.join(program);

        if candidate.is_file() {
            return Some(candidate);
        }
    }

    None
}

fn pass_cmake_env_vars(config: &mut Config) {
    for (key, value) in env::vars() {
        if key.starts_with("CMAKE_") {
            config.define(&key, &value);
        }
    }
}

fn configure_cpu_features(config: &mut Config, target_triple: &str) {
    let target_cpu = env::var("CARGO_ENCODED_RUSTFLAGS")
        .ok()
        .and_then(|rustflags| {
            rustflags
                .split('\x1f')
                .find(|flag| flag.contains("target-cpu="))
                .and_then(|flag| flag.split("target-cpu=").nth(1))
                .map(std::string::ToString::to_string)
        });

    if target_cpu.as_deref() == Some("native") {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");

        return;
    }

    config.define("GGML_NATIVE", "OFF");

    if let Some(ref cpu) = target_cpu {
        debug_log!("Setting baseline architecture: -march={}", cpu);
        config.cflag(format!("-march={cpu}"));
        config.cxxflag(format!("-march={cpu}"));
    }

    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    debug_log!("Compiling with target features: {}", features);

    for feature in features.split(',') {
        if let Some(ggml_flag) = map_cpu_feature_to_ggml(feature) {
            config.define(ggml_flag, "ON");
        }
    }

    if target_triple.contains("aarch64")
        && target_triple.contains("linux")
        && target_cpu.as_deref() != Some("native")
    {
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }
}

fn map_cpu_feature_to_ggml(feature: &str) -> Option<&'static str> {
    match feature {
        "avx" => Some("GGML_AVX"),
        "avx2" => Some("GGML_AVX2"),
        "avx512bf16" => Some("GGML_AVX512_BF16"),
        "avx512vbmi" => Some("GGML_AVX512_VBMI"),
        "avx512vnni" => Some("GGML_AVX512_VNNI"),
        "avxvnni" => Some("GGML_AVX_VNNI"),
        "bmi2" => Some("GGML_BMI2"),
        "f16c" => Some("GGML_F16C"),
        "fma" => Some("GGML_FMA"),
        "sse4.2" => Some("GGML_SSE42"),
        _ => {
            debug_log!(
                "Unrecognized cpu feature: '{}' - skipping GGML config for it.",
                feature
            );

            None
        }
    }
}

fn configure_shared_libs(config: &mut Config, build_shared_libs: bool) {
    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );
}

fn configure_platform_specific(
    config: &mut Config,
    target_os: &TargetOs,
    target_triple: &str,
    profile: &str,
    android_ndk: Option<&AndroidNdk>,
) {
    match target_os {
        TargetOs::Apple(_) => {
            config.define("GGML_BLAS", "OFF");
            override_archive_commands_for_apple_ar(config);
        }
        TargetOs::Windows(WindowsVariant::Msvc) => {
            config.cflag("/w");
            config.cxxflag("/w");
            config.cxxflag("/EHsc");
            configure_msvc_release_workaround(config, profile);
        }
        TargetOs::Android => {
            if let Some(ndk) = android_ndk {
                configure_android_cmake(config, ndk, target_triple);
            }
        }
        _ => {}
    }
}

fn configure_msvc_release_workaround(config: &mut Config, profile: &str) {
    let is_release_profile = matches!(profile, "Release" | "RelWithDebInfo" | "MinSizeRel");

    if !is_release_profile {
        return;
    }

    for flag in &["/O2", "/DNDEBUG", "/Ob2"] {
        config.cflag(flag);
        config.cxxflag(flag);
    }
}

fn configure_android_cmake(config: &mut Config, ndk: &AndroidNdk, _target_triple: &str) {
    #[cfg(all(feature = "shared-stdcxx", feature = "static-stdcxx"))]
    compile_error!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");

    println!("cargo:rerun-if-env-changed=ANDROID_NDK");
    println!("cargo:rerun-if-env-changed=NDK_ROOT");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=ANDROID_PLATFORM");
    println!("cargo:rerun-if-env-changed=ANDROID_API_LEVEL");

    config.define("CMAKE_TOOLCHAIN_FILE", ndk.cmake_toolchain_file());
    config.define("ANDROID_PLATFORM", ndk.android_platform());
    config.define("ANDROID_ABI", ndk.abi);

    if cfg!(feature = "static-stdcxx") {
        config.define("ANDROID_STL", "c++_static");
    } else if cfg!(feature = "shared-stdcxx") {
        config.define("ANDROID_STL", "c++_shared");
    }

    configure_android_arch_flags(config, ndk.abi);

    config.define("GGML_LLAMAFILE", "OFF");

    println!("cargo:rustc-link-lib=log");
    println!("cargo:rustc-link-lib=android");
}

fn override_archive_commands_for_apple_ar(config: &mut Config) {
    for language in ["C", "CXX", "OBJC", "OBJCXX"] {
        config.define(
            format!("CMAKE_{language}_ARCHIVE_CREATE"),
            "<CMAKE_AR> qc <TARGET> <LINK_FLAGS> <OBJECTS>",
        );
        config.define(
            format!("CMAKE_{language}_ARCHIVE_APPEND"),
            "<CMAKE_AR> q <TARGET> <LINK_FLAGS> <OBJECTS>",
        );
        config.define(
            format!("CMAKE_{language}_ARCHIVE_FINISH"),
            "<CMAKE_RANLIB> <TARGET>",
        );
    }
}

fn configure_android_arch_flags(config: &mut Config, abi: &str) {
    match abi {
        "arm64-v8a" => {
            config.cflag("-march=armv8-a");
            config.cxxflag("-march=armv8-a");
        }
        "armeabi-v7a" => {
            config.cflag("-march=armv7-a");
            config.cxxflag("-march=armv7-a");
            config.cflag("-mfpu=neon");
            config.cxxflag("-mfpu=neon");
            config.cflag("-mthumb");
            config.cxxflag("-mthumb");
        }
        "x86_64" => {
            config.cflag("-march=x86-64");
            config.cxxflag("-march=x86-64");
        }
        "x86" => {
            config.cflag("-march=i686");
            config.cxxflag("-march=i686");
        }
        _ => {}
    }
}

fn configure_gpu_backends(config: &mut Config, target_os: &TargetOs) {
    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        configure_vulkan_linking(config, target_os);
    }

    if cfg!(feature = "cuda") {
        config.define("GGML_CUDA", "ON");

        if cfg!(feature = "cuda-no-vmm") {
            config.define("GGML_CUDA_NO_VMM", "ON");
        }
    }

    if cfg!(feature = "rocm") {
        config.define("GGML_HIP", "ON");
    }
}

fn configure_vulkan_linking(config: &mut Config, target_os: &TargetOs) {
    match target_os {
        TargetOs::Windows(_) => {
            let vulkan_path = env::var("VULKAN_SDK")
                .expect("Please install Vulkan SDK and ensure that VULKAN_SDK env variable is set");
            let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");

            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");

            // SAFETY: build scripts are single-threaded, so modifying env is safe.
            unsafe { env::set_var("TrackFileAccess", "false") };

            config.cflag("/FS");
            config.cxxflag("/FS");
        }
        TargetOs::Linux => {
            if let Ok(vulkan_path) = env::var("VULKAN_SDK") {
                let vulkan_lib_path = Path::new(&vulkan_path).join("lib");

                println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            }

            println!("cargo:rustc-link-lib=vulkan");
        }
        _ => (),
    }
}

fn configure_openmp(config: &mut Config, target_os: &TargetOs) {
    let openmp_enabled = cfg!(feature = "openmp") && !target_os.is_android();

    config.define("GGML_OPENMP", if openmp_enabled { "ON" } else { "OFF" });
}

fn configure_system_ggml(config: &mut Config) {
    if cfg!(feature = "system-ggml") {
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
    }
}

#[cfg(test)]
mod tests {
    use cmake::Config;
    use serial_test::serial;

    use crate::android_ndk::AndroidNdk;
    use crate::scratch_dir::ScratchDir;
    use crate::target_os::TargetOs;

    use super::configure_android_arch_flags;
    use super::configure_android_cmake;
    use super::configure_base_defines;
    use super::configure_compiler_launchers;
    use super::configure_cpu_features;
    use super::configure_dynamic_backends;
    use super::configure_gpu_backends;
    use super::configure_msvc_release_workaround;
    use super::configure_openmp;
    use super::configure_platform_specific;
    use super::configure_shared_libs;
    use super::configure_system_ggml;
    use super::configure_vulkan_linking;
    use super::map_cpu_feature_to_ggml;
    use super::override_archive_commands_for_apple_ar;
    use super::pass_cmake_env_vars;
    use super::which;

    fn config() -> Config {
        Config::new(".")
    }

    fn target_os(triple: &str) -> TargetOs {
        TargetOs::from_target_triple(triple).expect("supported triple")
    }

    fn android_ndk(abi: &'static str) -> AndroidNdk {
        AndroidNdk {
            ndk_path: "/ndk".to_owned(),
            api_level: "28".to_owned(),
            abi,
            host_tag: "darwin-x86_64",
            toolchain_path: "/ndk/toolchain".to_owned(),
            sysroot: "/ndk/toolchain/sysroot".to_owned(),
            target_prefix: "aarch64-linux-android",
            clang_builtin_includes: None,
        }
    }

    #[test]
    fn every_recognised_cpu_feature_maps_to_a_ggml_flag() {
        for (feature, flag) in [
            ("avx", "GGML_AVX"),
            ("avx2", "GGML_AVX2"),
            ("avx512bf16", "GGML_AVX512_BF16"),
            ("avx512vbmi", "GGML_AVX512_VBMI"),
            ("avx512vnni", "GGML_AVX512_VNNI"),
            ("avxvnni", "GGML_AVX_VNNI"),
            ("bmi2", "GGML_BMI2"),
            ("f16c", "GGML_F16C"),
            ("fma", "GGML_FMA"),
            ("sse4.2", "GGML_SSE42"),
        ] {
            assert_eq!(map_cpu_feature_to_ggml(feature), Some(flag));
        }

        assert_eq!(map_cpu_feature_to_ggml("neon"), None);
    }

    #[test]
    #[serial]
    fn which_finds_a_program_on_path_and_reports_absence() {
        let scratch = ScratchDir::new("cmake-which");
        let program = scratch.path().join("a-build-tool");
        std::fs::write(&program, b"#!/bin/sh\n").expect("program must be writable");
        let previous = std::env::var_os("PATH").expect("PATH is always set");
        unsafe { std::env::set_var("PATH", scratch.path()) };

        let found = which("a-build-tool");
        let missing = which("definitely-not-a-program");

        unsafe { std::env::set_var("PATH", previous) };

        assert_eq!(found, Some(program));
        assert_eq!(missing, None);
    }

    #[test]
    #[serial]
    fn dynamic_backends_are_configured_only_behind_the_feature() {
        let scratch = ScratchDir::new("cmake-backends");
        let mut config = config();

        let backends_dir = configure_dynamic_backends(&mut config, scratch.path());

        if cfg!(feature = "dynamic-backends") {
            let dir = backends_dir.expect("the feature must yield a backends directory");
            assert!(dir.is_dir(), "the backends directory must be created");
        } else {
            assert_eq!(backends_dir, None);
        }
    }

    #[test]
    #[serial]
    fn compiler_launchers_are_skipped_when_ccache_is_disabled() {
        unsafe { std::env::set_var("LLAMA_DISABLE_CCACHE", "1") };

        configure_compiler_launchers(&mut config());

        unsafe { std::env::remove_var("LLAMA_DISABLE_CCACHE") };
    }

    #[test]
    #[serial]
    fn compiler_launchers_are_configured_when_ccache_is_present() {
        let scratch = ScratchDir::new("cmake-ccache");
        std::fs::write(scratch.path().join("ccache"), b"#!/bin/sh\n")
            .expect("ccache stand-in must be writable");
        let previous_path = std::env::var_os("PATH").expect("PATH is always set");

        unsafe {
            std::env::remove_var("LLAMA_DISABLE_CCACHE");
            std::env::set_var("PATH", scratch.path());
        }

        configure_compiler_launchers(&mut config());

        unsafe { std::env::set_var("PATH", previous_path) };
    }

    #[test]
    #[serial]
    fn cmake_prefixed_environment_variables_are_forwarded() {
        unsafe { std::env::set_var("CMAKE_A_TEST_ONLY_VARIABLE", "value") };

        pass_cmake_env_vars(&mut config());

        unsafe { std::env::remove_var("CMAKE_A_TEST_ONLY_VARIABLE") };
    }

    #[test]
    #[serial]
    fn a_native_target_cpu_enables_ggml_native() {
        unsafe {
            std::env::set_var("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native");
        }

        configure_cpu_features(&mut config(), "aarch64-apple-darwin");

        unsafe { std::env::remove_var("CARGO_ENCODED_RUSTFLAGS") };
    }

    #[test]
    #[serial]
    fn an_explicit_target_cpu_sets_a_baseline_march() {
        unsafe {
            std::env::set_var("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=haswell");
            std::env::set_var("CARGO_CFG_TARGET_FEATURE", "avx,avx2,neon");
        }

        configure_cpu_features(&mut config(), "aarch64-unknown-linux-gnu");

        unsafe {
            std::env::remove_var("CARGO_ENCODED_RUSTFLAGS");
            std::env::remove_var("CARGO_CFG_TARGET_FEATURE");
        }
    }

    #[test]
    #[serial]
    fn an_absent_target_cpu_still_configures_features() {
        unsafe {
            std::env::remove_var("CARGO_ENCODED_RUSTFLAGS");
            std::env::remove_var("CARGO_CFG_TARGET_FEATURE");
        }

        configure_cpu_features(&mut config(), "x86_64-unknown-linux-gnu");
    }

    #[test]
    fn shared_library_configuration_covers_both_settings() {
        configure_shared_libs(&mut config(), true);
        configure_shared_libs(&mut config(), false);
    }

    #[test]
    fn every_platform_arm_is_configured() {
        configure_platform_specific(
            &mut config(),
            &target_os("aarch64-apple-darwin"),
            "aarch64-apple-darwin",
            "Release",
            None,
        );
        configure_platform_specific(
            &mut config(),
            &target_os("x86_64-pc-windows-msvc"),
            "x86_64-pc-windows-msvc",
            "Release",
            None,
        );
        configure_platform_specific(
            &mut config(),
            &target_os("x86_64-unknown-linux-gnu"),
            "x86_64-unknown-linux-gnu",
            "Release",
            None,
        );
        configure_platform_specific(
            &mut config(),
            &target_os("aarch64-linux-android"),
            "aarch64-linux-android",
            "Release",
            None,
        );
        configure_platform_specific(
            &mut config(),
            &target_os("aarch64-linux-android"),
            "aarch64-linux-android",
            "Release",
            Some(&android_ndk("arm64-v8a")),
        );
    }

    #[test]
    fn the_msvc_release_workaround_applies_only_to_release_profiles() {
        for profile in ["Release", "RelWithDebInfo", "MinSizeRel", "Debug"] {
            configure_msvc_release_workaround(&mut config(), profile);
        }
    }

    #[test]
    fn the_android_cmake_configuration_uses_the_detected_ndk() {
        configure_android_cmake(
            &mut config(),
            &android_ndk("arm64-v8a"),
            "aarch64-linux-android",
        );
    }

    #[test]
    fn every_android_abi_sets_architecture_flags() {
        for abi in ["arm64-v8a", "armeabi-v7a", "x86_64", "x86", "riscv64"] {
            configure_android_arch_flags(&mut config(), abi);
        }
    }

    #[test]
    fn apple_archive_commands_are_overridden() {
        override_archive_commands_for_apple_ar(&mut config());
    }

    #[test]
    fn gpu_backends_are_configured_for_each_host() {
        configure_gpu_backends(&mut config(), &target_os("aarch64-apple-darwin"));
        configure_gpu_backends(&mut config(), &target_os("x86_64-unknown-linux-gnu"));
    }

    #[test]
    #[serial]
    fn vulkan_linking_covers_every_platform_arm() {
        let scratch = ScratchDir::new("cmake-vulkan");
        unsafe { std::env::set_var("VULKAN_SDK", scratch.path()) };

        configure_vulkan_linking(&mut config(), &target_os("x86_64-pc-windows-msvc"));
        configure_vulkan_linking(&mut config(), &target_os("x86_64-unknown-linux-gnu"));
        configure_vulkan_linking(&mut config(), &target_os("aarch64-apple-darwin"));

        unsafe { std::env::remove_var("VULKAN_SDK") };

        configure_vulkan_linking(&mut config(), &target_os("x86_64-unknown-linux-gnu"));

        assert_eq!(
            std::env::var("TrackFileAccess").as_deref(),
            Ok("false"),
            "the windows arm must disable MSBuild file tracking"
        );

        unsafe { std::env::remove_var("TrackFileAccess") };
    }

    #[test]
    fn openmp_is_disabled_for_android_and_configured_elsewhere() {
        configure_openmp(&mut config(), &target_os("aarch64-linux-android"));
        configure_openmp(&mut config(), &target_os("aarch64-apple-darwin"));
    }

    #[test]
    fn system_ggml_configuration_runs() {
        configure_system_ggml(&mut config());
    }

    #[test]
    fn the_base_defines_turn_off_upstream_extras() {
        configure_base_defines(&mut config());
    }

    /// Drives the whole configure-and-build path against a minimal CMake
    /// project, so the orchestration is exercised without building llama.cpp.
    #[test]
    #[serial]
    fn a_minimal_project_is_configured_and_built() {
        let scratch = ScratchDir::new("cmake-build");
        let source_dir = scratch.path().join("source");
        std::fs::create_dir_all(&source_dir).expect("source dir must be creatable");
        std::fs::write(
            source_dir.join("CMakeLists.txt"),
            b"cmake_minimum_required(VERSION 3.14)\n\
              project(probe C)\n\
              add_library(probe STATIC probe.c)\n\
              install(TARGETS probe ARCHIVE DESTINATION lib)\n",
        )
        .expect("CMakeLists must be writable");
        std::fs::write(
            source_dir.join("probe.c"),
            b"int probe(void) { return 1; }\n",
        )
        .expect("source must be writable");

        let context = crate::BuildContext {
            manifest_dir: scratch.path().to_path_buf(),
            out_dir: scratch.path().join("out"),
            target_dir: scratch.path().join("target"),
            cmake_dir: scratch.path().join("cmake-out"),
            llama_src: source_dir,
            target_os: target_os("aarch64-apple-darwin"),
            target_triple: "aarch64-apple-darwin".to_owned(),
            build_shared_libs: false,
            profile: "Release".to_owned(),
            static_crt: false,
            android_ndk: None,
        };

        let install_dir =
            crate::cc_test_environment::with_cc_environment_value(scratch.path(), || {
                super::configure_and_build(&context)
            });

        assert!(
            install_dir.join("lib").join("libprobe.a").exists(),
            "the archive must be installed under the cmake out dir"
        );
    }
}
