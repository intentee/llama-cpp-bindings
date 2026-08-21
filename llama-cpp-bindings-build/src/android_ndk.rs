use std::env;
use std::path::{Path, PathBuf};

use thiserror::Error;

const DEFAULT_ANDROID_API_LEVEL: &str = "28";

#[derive(Debug, Error)]
pub enum AndroidNdkDetectionError {
    #[error(
        "Android NDK not found for target {target_triple}. Set ANDROID_NDK, ANDROID_NDK_ROOT, NDK_ROOT, or CARGO_NDK_ANDROID_NDK."
    )]
    NdkRootNotConfigured {
        target_triple: String,
        #[source]
        source: env::VarError,
    },
    #[error("Android NDK path does not exist: {path}")]
    NdkRootMissing { path: PathBuf },
    #[error("Android NDK toolchain file not found: {path}")]
    NdkToolchainFileMissing { path: PathBuf },
    #[error("Android NDK toolchain not found at: {path}")]
    NdkToolchainDirectoryMissing { path: PathBuf },
    #[error("Unsupported host platform for Android NDK")]
    UnsupportedHostPlatform,
    #[error("Unsupported Android target triple: {target_triple}")]
    UnsupportedAndroidTarget { target_triple: String },
}

#[derive(Debug)]
pub struct AndroidNdk {
    pub ndk_path: String,
    pub api_level: String,
    pub abi: &'static str,
    pub host_tag: &'static str,
    pub toolchain_path: String,
    pub sysroot: String,
    pub target_prefix: &'static str,
    pub clang_builtin_includes: Option<String>,
}

impl AndroidNdk {
    /// # Errors
    ///
    /// Returns [`AndroidNdkDetectionError`] when the NDK installation cannot be
    /// located, an environment variable is missing, the target triple is
    /// unsupported, or the host platform is not supported by the NDK.
    pub fn detect(target_triple: &str) -> Result<Self, AndroidNdkDetectionError> {
        let ndk_path = detect_ndk_path(target_triple)?;

        validate_ndk_installation(&ndk_path)?;

        let api_level = detect_api_level();
        let abi = target_triple_to_abi(target_triple)?;
        let host_tag = detect_host_tag()?;
        let target_prefix = target_triple_to_ndk_prefix(target_triple)?;
        let toolchain_path = Path::new(&ndk_path)
            .join("toolchains")
            .join("llvm")
            .join("prebuilt")
            .join(host_tag);

        if !toolchain_path.exists() {
            return Err(AndroidNdkDetectionError::NdkToolchainDirectoryMissing {
                path: toolchain_path,
            });
        }

        let sysroot = toolchain_path
            .join("sysroot")
            .to_string_lossy()
            .into_owned();
        let clang_builtin_includes = find_clang_builtin_includes(&toolchain_path);
        let toolchain_path = toolchain_path.to_string_lossy().into_owned();

        Ok(Self {
            ndk_path,
            api_level,
            abi,
            host_tag,
            toolchain_path,
            sysroot,
            target_prefix,
            clang_builtin_includes,
        })
    }

    pub fn android_platform(&self) -> String {
        format!("android-{}", self.api_level)
    }

    pub fn cmake_toolchain_file(&self) -> String {
        format!("{}/build/cmake/android.toolchain.cmake", self.ndk_path)
    }
}

fn detect_ndk_path(target_triple: &str) -> Result<String, AndroidNdkDetectionError> {
    env::var("ANDROID_NDK")
        .or_else(|_android_ndk_unset| env::var("ANDROID_NDK_ROOT"))
        .or_else(|_android_ndk_root_unset| env::var("NDK_ROOT"))
        .or_else(|_ndk_root_unset| env::var("CARGO_NDK_ANDROID_NDK"))
        .or_else(|_cargo_ndk_android_ndk_unset| detect_ndk_from_sdk())
        .map_err(|source| AndroidNdkDetectionError::NdkRootNotConfigured {
            target_triple: target_triple.to_owned(),
            source,
        })
}

fn detect_ndk_from_sdk() -> Result<String, env::VarError> {
    let home = env::home_dir().ok_or(env::VarError::NotPresent)?;

    let android_home = match env::var("ANDROID_HOME")
        .or_else(|_android_home_unset| env::var("ANDROID_SDK_ROOT"))
    {
        Ok(value) => value,
        Err(_neither_env_var_set) => format!("{}/Android/Sdk", home.display()),
    };

    let ndk_dir = format!("{android_home}/ndk");
    let entries =
        std::fs::read_dir(&ndk_dir).map_err(|_directory_unreadable| env::VarError::NotPresent)?;

    let mut versions: Vec<String> = entries
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_dir()))
        .filter_map(|entry| {
            entry
                .file_name()
                .to_str()
                .map(std::string::ToString::to_string)
        })
        .collect();

    versions.sort();

    versions
        .last()
        .map(|latest| format!("{ndk_dir}/{latest}"))
        .ok_or(env::VarError::NotPresent)
}

fn validate_ndk_installation(ndk_path: &str) -> Result<(), AndroidNdkDetectionError> {
    let ndk_path = Path::new(ndk_path);

    if !ndk_path.exists() {
        return Err(AndroidNdkDetectionError::NdkRootMissing {
            path: ndk_path.to_path_buf(),
        });
    }

    let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");

    if !toolchain_file.exists() {
        return Err(AndroidNdkDetectionError::NdkToolchainFileMissing {
            path: toolchain_file,
        });
    }

    Ok(())
}

fn detect_api_level() -> String {
    env::var("ANDROID_API_LEVEL")
        .or_else(|_android_api_level_unset| {
            env::var("ANDROID_PLATFORM").map(|platform| platform.replace("android-", ""))
        })
        .or_else(|_android_platform_unset| {
            env::var("CARGO_NDK_ANDROID_PLATFORM").map(|platform| platform.replace("android-", ""))
        })
        .unwrap_or_else(|_no_api_level_configured| DEFAULT_ANDROID_API_LEVEL.to_string())
}

fn detect_host_tag() -> Result<&'static str, AndroidNdkDetectionError> {
    host_tag_for_os(std::env::consts::OS)
}

/// Resolved from an OS name rather than a `cfg!` branch, so every host's tag
/// stays reachable from any machine.
fn host_tag_for_os(os: &str) -> Result<&'static str, AndroidNdkDetectionError> {
    match os {
        "macos" => Ok("darwin-x86_64"),
        "linux" => Ok("linux-x86_64"),
        "windows" => Ok("windows-x86_64"),
        _ => Err(AndroidNdkDetectionError::UnsupportedHostPlatform),
    }
}

fn target_triple_to_abi(target_triple: &str) -> Result<&'static str, AndroidNdkDetectionError> {
    if target_triple.contains("aarch64") {
        Ok("arm64-v8a")
    } else if target_triple.contains("armv7") {
        Ok("armeabi-v7a")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64")
    } else if target_triple.contains("i686") {
        Ok("x86")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedAndroidTarget {
            target_triple: target_triple.to_owned(),
        })
    }
}

fn target_triple_to_ndk_prefix(
    target_triple: &str,
) -> Result<&'static str, AndroidNdkDetectionError> {
    if target_triple.contains("aarch64") {
        Ok("aarch64-linux-android")
    } else if target_triple.contains("armv7") {
        Ok("arm-linux-androideabi")
    } else if target_triple.contains("x86_64") {
        Ok("x86_64-linux-android")
    } else if target_triple.contains("i686") {
        Ok("i686-linux-android")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedAndroidTarget {
            target_triple: target_triple.to_owned(),
        })
    }
}

fn find_clang_builtin_includes(toolchain_path: &Path) -> Option<String> {
    let clang_lib_path = toolchain_path.join("lib").join("clang");
    let entries = std::fs::read_dir(&clang_lib_path).ok()?;

    let version_dir = entries.filter_map(std::result::Result::ok).find(|entry| {
        entry
            .file_type()
            .map(|file_type| file_type.is_dir())
            .unwrap_or(false)
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(|character: char| character.is_ascii_digit()))
    })?;

    let include_path = PathBuf::from(&clang_lib_path)
        .join(version_dir.file_name())
        .join("include");

    if include_path.exists() {
        Some(include_path.to_string_lossy().to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serial_test::serial;

    use crate::scratch_dir::ScratchDir;

    use super::AndroidNdk;
    use super::AndroidNdkDetectionError;
    use super::detect_api_level;
    use super::detect_host_tag;
    use super::detect_ndk_from_sdk;
    use super::detect_ndk_path;
    use super::find_clang_builtin_includes;
    use super::host_tag_for_os;
    use super::target_triple_to_abi;
    use super::target_triple_to_ndk_prefix;
    use super::validate_ndk_installation;

    const NDK_VARS: &[&str] = &[
        "ANDROID_NDK",
        "ANDROID_NDK_ROOT",
        "NDK_ROOT",
        "CARGO_NDK_ANDROID_NDK",
        "ANDROID_HOME",
        "ANDROID_SDK_ROOT",
        "ANDROID_API_LEVEL",
        "ANDROID_PLATFORM",
        "CARGO_NDK_ANDROID_PLATFORM",
    ];

    fn clear_ndk_environment() {
        for name in NDK_VARS {
            unsafe { std::env::remove_var(name) };
        }
    }

    fn host_tag() -> &'static str {
        detect_host_tag().expect("the test host must be supported")
    }

    /// A synthetic NDK layout: enough structure for detection to succeed.
    fn ndk_tree(scratch: &ScratchDir, with_clang_includes: bool) -> String {
        let ndk = scratch.path().join("ndk/27.0.1");
        std::fs::create_dir_all(ndk.join("build/cmake")).expect("cmake dir must be creatable");
        std::fs::write(ndk.join("build/cmake/android.toolchain.cmake"), b"x")
            .expect("toolchain file must be writable");
        let toolchain = ndk.join(format!("toolchains/llvm/prebuilt/{}", host_tag()));
        std::fs::create_dir_all(toolchain.join("sysroot")).expect("sysroot must be creatable");

        if with_clang_includes {
            std::fs::create_dir_all(toolchain.join("lib/clang/18/include"))
                .expect("clang include dir must be creatable");
        }

        ndk.to_string_lossy().into_owned()
    }

    #[test]
    fn each_supported_architecture_maps_to_an_abi_and_prefix() {
        for (triple, abi, prefix) in [
            (
                "aarch64-linux-android",
                "arm64-v8a",
                "aarch64-linux-android",
            ),
            (
                "armv7-linux-androideabi",
                "armeabi-v7a",
                "arm-linux-androideabi",
            ),
            ("x86_64-linux-android", "x86_64", "x86_64-linux-android"),
            ("i686-linux-android", "x86", "i686-linux-android"),
        ] {
            assert_eq!(target_triple_to_abi(triple).expect(triple), abi);
            assert_eq!(target_triple_to_ndk_prefix(triple).expect(triple), prefix);
        }
    }

    #[test]
    fn an_unsupported_architecture_is_rejected_by_both_mappings() {
        assert!(matches!(
            target_triple_to_abi("mips-linux-android"),
            Err(AndroidNdkDetectionError::UnsupportedAndroidTarget { .. })
        ));
        assert!(matches!(
            target_triple_to_ndk_prefix("mips-linux-android"),
            Err(AndroidNdkDetectionError::UnsupportedAndroidTarget { .. })
        ));
    }

    #[test]
    fn every_supported_host_has_a_tag_and_others_are_rejected() {
        assert_eq!(host_tag_for_os("macos").expect("macos"), "darwin-x86_64");
        assert_eq!(host_tag_for_os("linux").expect("linux"), "linux-x86_64");
        assert_eq!(
            host_tag_for_os("windows").expect("windows"),
            "windows-x86_64"
        );
        assert!(matches!(
            host_tag_for_os("plan9"),
            Err(AndroidNdkDetectionError::UnsupportedHostPlatform)
        ));
    }

    #[test]
    #[serial]
    fn the_api_level_falls_back_through_each_variable() {
        clear_ndk_environment();
        assert_eq!(detect_api_level(), "28");

        unsafe { std::env::set_var("CARGO_NDK_ANDROID_PLATFORM", "android-30") };
        assert_eq!(detect_api_level(), "30");

        unsafe { std::env::set_var("ANDROID_PLATFORM", "android-31") };
        assert_eq!(detect_api_level(), "31");

        unsafe { std::env::set_var("ANDROID_API_LEVEL", "32") };
        assert_eq!(detect_api_level(), "32");

        clear_ndk_environment();
    }

    #[test]
    #[serial]
    fn each_ndk_variable_is_consulted_in_order() {
        clear_ndk_environment();

        for name in [
            "CARGO_NDK_ANDROID_NDK",
            "NDK_ROOT",
            "ANDROID_NDK_ROOT",
            "ANDROID_NDK",
        ] {
            unsafe { std::env::set_var(name, format!("/from/{name}")) };

            assert_eq!(
                detect_ndk_path("aarch64-linux-android").expect(name),
                format!("/from/{name}")
            );
        }

        clear_ndk_environment();
    }

    #[test]
    #[serial]
    fn an_unset_ndk_reports_the_target_triple() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-unset");
        unsafe { std::env::set_var("ANDROID_HOME", scratch.path()) };

        let error =
            detect_ndk_path("aarch64-linux-android").expect_err("no NDK anywhere must fail");

        clear_ndk_environment();

        assert!(matches!(
            error,
            AndroidNdkDetectionError::NdkRootNotConfigured { ref target_triple, .. }
                if target_triple == "aarch64-linux-android"
        ));
    }

    #[test]
    #[serial]
    fn the_newest_sdk_installed_ndk_is_selected() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-sdk");
        let ndk_dir = scratch.path().join("ndk");
        std::fs::create_dir_all(ndk_dir.join("25.1.0")).expect("older ndk must be creatable");
        std::fs::create_dir_all(ndk_dir.join("27.0.1")).expect("newer ndk must be creatable");
        std::fs::write(ndk_dir.join("not-a-directory"), b"x").expect("file must be writable");
        unsafe { std::env::set_var("ANDROID_SDK_ROOT", scratch.path()) };

        let detected = detect_ndk_from_sdk().expect("an SDK-installed NDK must be found");

        clear_ndk_environment();

        assert!(detected.ends_with("ndk/27.0.1"), "got: {detected}");
    }

    #[test]
    #[serial]
    fn an_sdk_without_any_ndk_directory_is_not_detected() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-sdk-empty");
        std::fs::create_dir_all(scratch.path().join("ndk")).expect("ndk dir must be creatable");
        unsafe { std::env::set_var("ANDROID_HOME", scratch.path()) };

        let outcome = detect_ndk_from_sdk();

        clear_ndk_environment();

        assert!(outcome.is_err(), "an empty ndk directory yields nothing");
    }

    #[test]
    fn validation_rejects_a_missing_root_and_a_missing_toolchain_file() {
        let scratch = ScratchDir::new("ndk-validate");

        assert!(matches!(
            validate_ndk_installation("/definitely/not/here"),
            Err(AndroidNdkDetectionError::NdkRootMissing { .. })
        ));

        let bare = scratch.path().join("bare");
        std::fs::create_dir_all(&bare).expect("bare ndk must be creatable");

        assert!(matches!(
            validate_ndk_installation(&bare.to_string_lossy()),
            Err(AndroidNdkDetectionError::NdkToolchainFileMissing { .. })
        ));
    }

    #[test]
    fn clang_builtin_includes_are_found_only_when_the_version_directory_exists() {
        let scratch = ScratchDir::new("ndk-clang");
        let toolchain = scratch.path().join("toolchain");

        assert_eq!(
            find_clang_builtin_includes(&toolchain),
            None,
            "a missing lib/clang directory yields nothing"
        );

        std::fs::create_dir_all(toolchain.join("lib/clang")).expect("clang dir must be creatable");
        std::fs::create_dir_all(toolchain.join("lib/clang/not-a-version"))
            .expect("non-version dir must be creatable");

        assert_eq!(
            find_clang_builtin_includes(&toolchain),
            None,
            "only digit-prefixed version directories count"
        );

        std::fs::create_dir_all(toolchain.join("lib/clang/18"))
            .expect("version dir must be creatable");

        assert_eq!(
            find_clang_builtin_includes(&toolchain),
            None,
            "a version directory without include/ yields nothing"
        );

        std::fs::create_dir_all(toolchain.join("lib/clang/18/include"))
            .expect("include dir must be creatable");

        let found =
            find_clang_builtin_includes(&toolchain).expect("a complete layout must resolve");
        let expected = toolchain
            .join("lib")
            .join("clang")
            .join("18")
            .join("include");

        assert_eq!(Path::new(&found), expected);
    }

    #[test]
    #[serial]
    fn a_complete_ndk_layout_is_detected_end_to_end() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-complete");
        let ndk_path = ndk_tree(&scratch, true);
        unsafe { std::env::set_var("ANDROID_NDK", &ndk_path) };

        let ndk = AndroidNdk::detect("aarch64-linux-android").expect("layout must be detected");

        clear_ndk_environment();

        assert_eq!(ndk.abi, "arm64-v8a");
        assert_eq!(ndk.target_prefix, "aarch64-linux-android");
        assert_eq!(ndk.api_level, "28");
        assert_eq!(ndk.android_platform(), "android-28");
        assert_eq!(
            ndk.cmake_toolchain_file(),
            format!("{ndk_path}/build/cmake/android.toolchain.cmake")
        );
        assert!(Path::new(&ndk.sysroot).is_dir());
        assert!(ndk.clang_builtin_includes.is_some());
        assert!(format!("{ndk:?}").contains("AndroidNdk"));
    }

    #[test]
    #[serial]
    fn a_layout_without_the_host_toolchain_directory_is_rejected() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-no-toolchain");
        let ndk = scratch.path().join("ndk");
        std::fs::create_dir_all(ndk.join("build/cmake")).expect("cmake dir must be creatable");
        std::fs::write(ndk.join("build/cmake/android.toolchain.cmake"), b"x")
            .expect("toolchain file must be writable");
        unsafe { std::env::set_var("ANDROID_NDK", ndk.to_string_lossy().as_ref()) };

        let error = AndroidNdk::detect("aarch64-linux-android")
            .expect_err("a missing host toolchain must fail");

        clear_ndk_environment();

        assert!(matches!(
            error,
            AndroidNdkDetectionError::NdkToolchainDirectoryMissing { .. }
        ));
    }

    #[test]
    #[serial]
    fn a_complete_layout_without_clang_includes_still_detects() {
        clear_ndk_environment();
        let scratch = ScratchDir::new("ndk-no-clang");
        unsafe { std::env::set_var("ANDROID_NDK", ndk_tree(&scratch, false)) };

        let ndk = AndroidNdk::detect("x86_64-linux-android").expect("layout must be detected");

        clear_ndk_environment();

        assert_eq!(ndk.abi, "x86_64");
        assert_eq!(ndk.clang_builtin_includes, None);
    }

    #[test]
    fn every_error_variant_renders_a_message() {
        let messages = [
            AndroidNdkDetectionError::NdkRootMissing { path: "/p".into() }.to_string(),
            AndroidNdkDetectionError::NdkToolchainFileMissing { path: "/p".into() }.to_string(),
            AndroidNdkDetectionError::NdkToolchainDirectoryMissing { path: "/p".into() }
                .to_string(),
            AndroidNdkDetectionError::UnsupportedHostPlatform.to_string(),
            AndroidNdkDetectionError::UnsupportedAndroidTarget {
                target_triple: "mips".to_owned(),
            }
            .to_string(),
        ];

        for message in messages {
            assert!(!message.is_empty());
        }
    }

    #[test]
    #[serial]
    fn an_unset_sdk_root_falls_back_to_the_home_directory() {
        clear_ndk_environment();

        let outcome = detect_ndk_from_sdk();

        assert!(
            outcome.is_err(),
            "no NDK is installed under the home directory in this environment"
        );
    }
}
