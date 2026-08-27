use std::env;
use std::path::{Path, PathBuf};

use cmake::Config;

use crate::BuildContext;
use crate::BuildError;
use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::optional_env;
use crate::target_os::TargetOs;

pub fn configure_and_build(context: &BuildContext) -> Result<PathBuf, BuildError> {
    let mut config = Config::new(&context.llama_src);

    configure_base_defines(&mut config);
    configure_cpu_features(
        &mut config,
        &context.cargo_cfg_target_arch,
        context.target_os,
    )?;
    configure_shared_libs(&mut config, context.build_shared_libs);
    configure_platform_specific(
        &mut config,
        context.target_os,
        &context.target_triple,
        context.android_ndk.as_ref(),
    );
    configure_gpu_backends(&mut config, context.target_os)?;
    configure_openmp(&mut config, context.target_os);
    configure_system_ggml(&mut config)?;
    let backends_dir = configure_dynamic_backends(&mut config, &context.cmake_dir)?;

    config.static_crt(context.static_crt);
    configure_msvc_exception_handling(&mut config, context.target_os);
    configure_msvc_config_flags(&mut config, context.target_os, &context.profile);
    config
        .out_dir(&context.cmake_dir)
        .profile(&context.profile)
        .always_configure(false);

    let install_dir = config.build();

    if let Some(dir) = backends_dir {
        println!("cargo:backends_dir={}", dir.display());
    }

    Ok(install_dir)
}

fn configure_dynamic_backends(
    config: &mut Config,
    cmake_dir: &Path,
) -> Result<Option<PathBuf>, BuildError> {
    if !cfg!(feature = "dynamic-backends") {
        return Ok(None);
    }

    let backends_dir = cmake_dir.join("backends");

    std::fs::create_dir_all(&backends_dir).map_err(|source| BuildError::Filesystem {
        path: backends_dir.clone(),
        source,
    })?;

    config.define("GGML_BACKEND_DL", "ON");
    config.define("GGML_CPU_ALL_VARIANTS", "ON");
    config.define("GGML_BACKEND_DIR", &backends_dir);

    Ok(Some(backends_dir))
}

fn configure_base_defines(config: &mut Config) {
    config.define("LLAMA_BUILD_TESTS", "OFF");
    config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    config.define("LLAMA_BUILD_SERVER", "OFF");
    config.define("LLAMA_BUILD_TOOLS", "OFF");
    config.define("LLAMA_BUILD_APP", "OFF");
    config.define("LLAMA_BUILD_COMMON", "ON");
    config.define("LLAMA_BUILD_MTMD", "ON");
    config.define("MTMD_VIDEO", "OFF");
    config.define("LLAMA_CURL", "OFF");
}

fn configure_cpu_features(
    config: &mut Config,
    cargo_cfg_target_arch: &str,
    target_os: TargetOs,
) -> Result<(), BuildError> {
    let target_cpu = optional_env("CARGO_ENCODED_RUSTFLAGS")?.and_then(|rustflags| {
        rustflags
            .split('\x1f')
            .find(|flag| flag.contains("target-cpu="))
            .and_then(|flag| flag.split("target-cpu=").nth(1))
            .map(ToString::to_string)
    });

    if target_cpu.as_deref() == Some("native") {
        debug_log!("Detected target-cpu=native, compiling with GGML_NATIVE");
        config.define("GGML_NATIVE", "ON");

        return Ok(());
    }

    config.define("GGML_NATIVE", "OFF");

    if let Some(ref cpu) = target_cpu {
        debug_log!("Setting baseline architecture: -march={}", cpu);
        config.cflag(format!("-march={cpu}"));
        config.cxxflag(format!("-march={cpu}"));
    }

    let features = optional_env("CARGO_CFG_TARGET_FEATURE")?.unwrap_or_default();
    debug_log!("Compiling with target features: {}", features);

    for feature in features.split(',') {
        if let Some(ggml_flag) = map_cpu_feature_to_ggml(feature) {
            config.define(ggml_flag, "ON");
        }
    }

    if cargo_cfg_target_arch == "aarch64"
        && target_os == TargetOs::Linux
        && target_cpu.as_deref() != Some("native")
    {
        config.define("GGML_CPU_ARM_ARCH", "armv8-a");
    }

    Ok(())
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
        _ => None,
    }
}

const fn msvc_exception_handling_flag(target_os: TargetOs) -> Option<&'static str> {
    if target_os.is_msvc() {
        Some("/EHsc")
    } else {
        None
    }
}

fn configure_msvc_exception_handling(config: &mut Config, target_os: TargetOs) {
    let Some(flag) = msvc_exception_handling_flag(target_os) else {
        return;
    };

    config.cxxflag(flag);
}

fn msvc_config_flags(target_os: TargetOs, profile: &str) -> Option<&'static str> {
    if !target_os.is_msvc() {
        return None;
    }

    match profile {
        "Debug" => Some("/Ob0 /Od /RTC1"),
        "MinSizeRel" => Some("/O1 /Ob1 /DNDEBUG"),
        "Release" => Some("/O2 /Ob2 /DNDEBUG"),
        "RelWithDebInfo" => Some("/O2 /Ob1 /DNDEBUG"),
        _ => None,
    }
}

fn configure_msvc_config_flags(config: &mut Config, target_os: TargetOs, profile: &str) {
    let Some(flags) = msvc_config_flags(target_os, profile) else {
        return;
    };
    let config_suffix = profile.to_uppercase();

    config.define(format!("CMAKE_C_FLAGS_{config_suffix}"), flags);
    config.define(format!("CMAKE_CXX_FLAGS_{config_suffix}"), flags);
}

fn configure_shared_libs(config: &mut Config, build_shared_libs: bool) {
    config.define(
        "BUILD_SHARED_LIBS",
        if build_shared_libs { "ON" } else { "OFF" },
    );
}

fn configure_platform_specific(
    config: &mut Config,
    target_os: TargetOs,
    target_triple: &str,
    android_ndk: Option<&AndroidNdk>,
) {
    match target_os {
        TargetOs::Apple(_) => {
            config.define("GGML_BLAS", "OFF");
        }
        TargetOs::Android => {
            if let Some(ndk) = android_ndk {
                configure_android_cmake(config, ndk, target_triple);
            }
        }
        _ => {}
    }
}

fn configure_android_cmake(config: &mut Config, ndk: &AndroidNdk, _target_triple: &str) {
    #[cfg(all(feature = "shared-stdcxx", feature = "static-stdcxx"))]
    compile_error!("Features 'shared-stdcxx' and 'static-stdcxx' are mutually exclusive");

    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_PLATFORM");

    config.define("CMAKE_TOOLCHAIN_FILE", ndk.cmake_toolchain_file());
    config.define("ANDROID_PLATFORM", ndk.android_platform());
    config.define("ANDROID_ABI", ndk.abi);

    if cfg!(feature = "static-stdcxx") {
        config.define("ANDROID_STL", "c++_static");
    } else if cfg!(feature = "shared-stdcxx") {
        config.define("ANDROID_STL", "c++_shared");
    }

    config.define("GGML_LLAMAFILE", "OFF");

    println!("cargo:rustc-link-lib=log");
    println!("cargo:rustc-link-lib=android");
}

fn configure_gpu_backends(config: &mut Config, target_os: TargetOs) -> Result<(), BuildError> {
    if cfg!(feature = "vulkan") {
        config.define("GGML_VULKAN", "ON");
        configure_vulkan_linking(target_os)?;
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

    Ok(())
}

fn configure_vulkan_linking(target_os: TargetOs) -> Result<(), BuildError> {
    match target_os {
        TargetOs::Windows(_) => {
            let vulkan_path = env::var("VULKAN_SDK").map_err(|source| BuildError::Environment {
                name: "VULKAN_SDK",
                source,
            })?;
            let vulkan_lib_path = Path::new(&vulkan_path).join("Lib");

            println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
            println!("cargo:rustc-link-lib=vulkan-1");
        }
        TargetOs::Linux => {
            match env::var("VULKAN_SDK") {
                Ok(vulkan_path) => {
                    let vulkan_lib_path = Path::new(&vulkan_path).join("lib");

                    println!("cargo:rustc-link-search={}", vulkan_lib_path.display());
                }
                Err(env::VarError::NotPresent) => {}
                Err(source) => {
                    return Err(BuildError::Environment {
                        name: "VULKAN_SDK",
                        source,
                    });
                }
            }

            println!("cargo:rustc-link-lib=vulkan");
        }
        _ => (),
    }

    Ok(())
}

fn configure_openmp(config: &mut Config, target_os: TargetOs) {
    let openmp_enabled = cfg!(feature = "openmp") && !target_os.is_android();

    config.define("GGML_OPENMP", if openmp_enabled { "ON" } else { "OFF" });
}

fn configure_system_ggml(config: &mut Config) -> Result<(), BuildError> {
    if cfg!(feature = "system-ggml") {
        println!("cargo:rerun-if-env-changed=GGML_DIR");
        let ggml_dir = env::var("GGML_DIR").map_err(|source| BuildError::Environment {
            name: "GGML_DIR",
            source,
        })?;
        config.define("LLAMA_USE_SYSTEM_GGML", "ON");
        config.define("GGML_DIR", ggml_dir);
    }

    Ok(())
}

#[cfg(test)]
mod msvc_exception_handling_tests {
    use crate::target_os::TargetOs;
    use crate::windows_variant::WindowsVariant;

    use super::msvc_exception_handling_flag;

    #[test]
    fn msvc_targets_compile_llama_cpp_with_unwind_semantics() {
        assert_eq!(
            msvc_exception_handling_flag(TargetOs::Windows(WindowsVariant::Msvc)),
            Some("/EHsc")
        );
    }

    #[test]
    fn targets_without_msvc_keep_their_toolchain_default_exception_handling() {
        assert_eq!(msvc_exception_handling_flag(TargetOs::Linux), None);
        assert_eq!(
            msvc_exception_handling_flag(TargetOs::Windows(WindowsVariant::Other)),
            None
        );
    }
}

#[cfg(test)]
mod msvc_config_flag_tests {
    use crate::target_os::TargetOs;
    use crate::windows_variant::WindowsVariant;

    use super::msvc_config_flags;

    #[test]
    fn every_msvc_configuration_keeps_llama_cpp_assertions_compiled_out() {
        let msvc = TargetOs::Windows(WindowsVariant::Msvc);

        assert_eq!(msvc_config_flags(msvc, "Debug"), Some("/Ob0 /Od /RTC1"));
        assert_eq!(
            msvc_config_flags(msvc, "MinSizeRel"),
            Some("/O1 /Ob1 /DNDEBUG")
        );
        assert_eq!(
            msvc_config_flags(msvc, "Release"),
            Some("/O2 /Ob2 /DNDEBUG")
        );
        assert_eq!(
            msvc_config_flags(msvc, "RelWithDebInfo"),
            Some("/O2 /Ob1 /DNDEBUG")
        );
    }

    #[test]
    fn targets_and_profiles_without_msvc_defaults_keep_the_flags_cmake_chose() {
        assert_eq!(msvc_config_flags(TargetOs::Linux, "Release"), None);
        assert_eq!(
            msvc_config_flags(TargetOs::Windows(WindowsVariant::Msvc), "Fastest"),
            None
        );
    }
}

#[cfg(test)]
mod cpu_feature_mapping_tests {
    use super::map_cpu_feature_to_ggml;

    #[test]
    fn every_supported_rust_cpu_feature_maps_to_its_ggml_switch() {
        assert_eq!(map_cpu_feature_to_ggml("avx"), Some("GGML_AVX"));
        assert_eq!(map_cpu_feature_to_ggml("avx2"), Some("GGML_AVX2"));
        assert_eq!(
            map_cpu_feature_to_ggml("avx512bf16"),
            Some("GGML_AVX512_BF16")
        );
        assert_eq!(
            map_cpu_feature_to_ggml("avx512vbmi"),
            Some("GGML_AVX512_VBMI")
        );
        assert_eq!(
            map_cpu_feature_to_ggml("avx512vnni"),
            Some("GGML_AVX512_VNNI")
        );
        assert_eq!(map_cpu_feature_to_ggml("avxvnni"), Some("GGML_AVX_VNNI"));
        assert_eq!(map_cpu_feature_to_ggml("bmi2"), Some("GGML_BMI2"));
        assert_eq!(map_cpu_feature_to_ggml("f16c"), Some("GGML_F16C"));
        assert_eq!(map_cpu_feature_to_ggml("fma"), Some("GGML_FMA"));
        assert_eq!(map_cpu_feature_to_ggml("sse4.2"), Some("GGML_SSE42"));
    }

    #[test]
    fn unsupported_rust_cpu_feature_does_not_enable_a_ggml_switch() {
        assert_eq!(map_cpu_feature_to_ggml("aes"), None);
        assert_eq!(map_cpu_feature_to_ggml(""), None);
    }
}
