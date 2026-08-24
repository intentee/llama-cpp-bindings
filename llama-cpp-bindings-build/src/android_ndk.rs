use std::env;
use std::path::{Path, PathBuf};

use thiserror::Error;

const DEFAULT_ANDROID_API_LEVEL: &str = "28";

#[derive(Debug, Error)]
pub enum AndroidNdkDetectionError {
    #[error("Android NDK not found for target {target_triple}. Set ANDROID_NDK_HOME.")]
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
    #[error("Android NDK Clang directory could not be read at {path}: {source}")]
    ClangDirectoryUnreadable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Android NDK Clang version directory was not found at {path}")]
    ClangVersionDirectoryMissing { path: PathBuf },
    #[error("Android NDK Clang directory entry could not be read at {path}: {source}")]
    ClangDirectoryEntryUnreadable {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("Android NDK Clang built-in include directory was not found at {path}")]
    ClangBuiltinIncludesMissing { path: PathBuf },
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
    pub clang_builtin_includes: String,
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
        let toolchain_path = format!("{ndk_path}/toolchains/llvm/prebuilt/{host_tag}");

        if !Path::new(&toolchain_path).exists() {
            return Err(AndroidNdkDetectionError::NdkToolchainDirectoryMissing {
                path: PathBuf::from(toolchain_path),
            });
        }

        let sysroot = format!("{toolchain_path}/sysroot");
        let clang_builtin_includes = find_clang_builtin_includes(&toolchain_path)?;

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
    env::var("ANDROID_NDK_HOME").map_err(|source| AndroidNdkDetectionError::NdkRootNotConfigured {
        target_triple: target_triple.to_owned(),
        source,
    })
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
    env::var("ANDROID_PLATFORM")
        .map(|platform| platform.replace("android-", ""))
        .unwrap_or_else(|_no_api_level_configured| DEFAULT_ANDROID_API_LEVEL.to_string())
}

fn detect_host_tag() -> Result<&'static str, AndroidNdkDetectionError> {
    if cfg!(target_os = "macos") {
        Ok("darwin-x86_64")
    } else if cfg!(target_os = "linux") {
        Ok("linux-x86_64")
    } else if cfg!(target_os = "windows") {
        Ok("windows-x86_64")
    } else {
        Err(AndroidNdkDetectionError::UnsupportedHostPlatform)
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

fn find_clang_builtin_includes(toolchain_path: &str) -> Result<String, AndroidNdkDetectionError> {
    let clang_lib_path = PathBuf::from(toolchain_path).join("lib/clang");
    let entries = std::fs::read_dir(&clang_lib_path).map_err(|source| {
        AndroidNdkDetectionError::ClangDirectoryUnreadable {
            path: clang_lib_path.clone(),
            source,
        }
    })?;
    let mut version_dir = None;
    for entry in entries {
        let entry =
            entry.map_err(
                |source| AndroidNdkDetectionError::ClangDirectoryEntryUnreadable {
                    path: clang_lib_path.clone(),
                    source,
                },
            )?;
        let file_type = entry.file_type().map_err(|source| {
            AndroidNdkDetectionError::ClangDirectoryEntryUnreadable {
                path: entry.path(),
                source,
            }
        })?;
        if file_type.is_dir()
            && entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with(|character: char| character.is_ascii_digit()))
        {
            version_dir = Some(entry);
            break;
        }
    }
    let version_dir =
        version_dir.ok_or_else(|| AndroidNdkDetectionError::ClangVersionDirectoryMissing {
            path: clang_lib_path.clone(),
        })?;

    let include_path = clang_lib_path.join(version_dir.file_name()).join("include");

    if !include_path.is_dir() {
        return Err(AndroidNdkDetectionError::ClangBuiltinIncludesMissing { path: include_path });
    }

    Ok(include_path.to_string_lossy().into_owned())
}

#[cfg(test)]
mod android_ndk_resolution_tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::AndroidNdk;
    use super::AndroidNdkDetectionError;
    use super::find_clang_builtin_includes;
    use super::target_triple_to_abi;
    use super::target_triple_to_ndk_prefix;
    use super::validate_ndk_installation;

    static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);

    fn temporary_directory(name: &str) -> PathBuf {
        let id = NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "llama-cpp-bindings-{name}-{}-{id}",
            std::process::id()
        ))
    }

    #[test]
    fn every_supported_android_target_maps_to_its_abi_and_ndk_prefix() {
        let targets = [
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
        ];

        for (target, abi, prefix) in targets {
            assert_eq!(target_triple_to_abi(target).expect("supported ABI"), abi);
            assert_eq!(
                target_triple_to_ndk_prefix(target).expect("supported NDK prefix"),
                prefix
            );
        }
    }

    #[test]
    fn unsupported_android_target_preserves_the_target_triple() {
        let abi_error =
            target_triple_to_abi("riscv64-linux-android").expect_err("unsupported ABI must fail");
        let prefix_error = target_triple_to_ndk_prefix("riscv64-linux-android")
            .expect_err("unsupported NDK prefix must fail");

        assert!(matches!(
            abi_error,
            AndroidNdkDetectionError::UnsupportedAndroidTarget { target_triple }
                if target_triple == "riscv64-linux-android"
        ));
        assert!(matches!(
            prefix_error,
            AndroidNdkDetectionError::UnsupportedAndroidTarget { target_triple }
                if target_triple == "riscv64-linux-android"
        ));
    }

    #[test]
    fn android_ndk_paths_derive_from_the_resolved_root_and_api_level() {
        let ndk = AndroidNdk {
            ndk_path: "/opt/android-ndk".to_owned(),
            api_level: "35".to_owned(),
            abi: "arm64-v8a",
            host_tag: "linux-x86_64",
            toolchain_path: "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64".to_owned(),
            sysroot: "/opt/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot".to_owned(),
            target_prefix: "aarch64-linux-android",
            clang_builtin_includes: "/opt/android-ndk/lib/clang/20/include".to_owned(),
        };

        assert_eq!(ndk.android_platform(), "android-35");
        assert_eq!(
            ndk.cmake_toolchain_file(),
            "/opt/android-ndk/build/cmake/android.toolchain.cmake"
        );
    }

    #[test]
    fn ndk_validation_rejects_a_missing_root_directory() {
        let root = temporary_directory("missing-ndk-root");

        let error = validate_ndk_installation(root.to_str().expect("UTF-8 temporary path"))
            .expect_err("missing NDK root must fail");

        assert!(matches!(
            error,
            AndroidNdkDetectionError::NdkRootMissing { path } if path == root
        ));
    }

    #[test]
    fn ndk_validation_rejects_a_root_without_a_cmake_toolchain() {
        let root = temporary_directory("ndk-without-toolchain");
        std::fs::create_dir_all(&root).expect("temporary NDK root must be created");

        let error = validate_ndk_installation(root.to_str().expect("UTF-8 temporary path"))
            .expect_err("missing toolchain file must fail");

        assert!(matches!(
            error,
            AndroidNdkDetectionError::NdkToolchainFileMissing { path }
                if path == root.join("build/cmake/android.toolchain.cmake")
        ));
        std::fs::remove_dir_all(root).expect("temporary NDK root must be removed");
    }

    #[test]
    fn ndk_validation_accepts_a_root_with_a_cmake_toolchain() {
        let root = temporary_directory("valid-ndk-root");
        let toolchain = root.join("build/cmake/android.toolchain.cmake");
        std::fs::create_dir_all(toolchain.parent().expect("toolchain parent"))
            .expect("toolchain directory must be created");
        std::fs::write(&toolchain, "").expect("toolchain file must be created");

        validate_ndk_installation(root.to_str().expect("UTF-8 temporary path"))
            .expect("valid NDK root must pass");

        std::fs::remove_dir_all(root).expect("temporary NDK root must be removed");
    }

    #[test]
    fn clang_builtin_include_resolution_rejects_a_missing_clang_directory() {
        let toolchain = temporary_directory("missing-clang-directory");

        let error = find_clang_builtin_includes(toolchain.to_str().expect("UTF-8 temporary path"))
            .expect_err("missing Clang directory must fail");

        assert!(matches!(
            error,
            AndroidNdkDetectionError::ClangDirectoryUnreadable { path, .. }
                if path == toolchain.join("lib/clang")
        ));
    }

    #[test]
    fn clang_builtin_include_resolution_requires_a_version_directory() {
        let toolchain = temporary_directory("clang-without-version");
        std::fs::create_dir_all(toolchain.join("lib/clang/not-a-version"))
            .expect("Clang directory must be created");

        let error = find_clang_builtin_includes(toolchain.to_str().expect("UTF-8 temporary path"))
            .expect_err("missing version directory must fail");

        assert!(matches!(
            error,
            AndroidNdkDetectionError::ClangVersionDirectoryMissing { path }
                if path == toolchain.join("lib/clang")
        ));
        std::fs::remove_dir_all(toolchain).expect("temporary toolchain must be removed");
    }

    #[test]
    fn clang_builtin_include_resolution_requires_an_include_directory() {
        let toolchain = temporary_directory("clang-without-includes");
        std::fs::create_dir_all(toolchain.join("lib/clang/20"))
            .expect("Clang version directory must be created");

        let error = find_clang_builtin_includes(toolchain.to_str().expect("UTF-8 temporary path"))
            .expect_err("missing built-in includes must fail");

        assert!(matches!(
            error,
            AndroidNdkDetectionError::ClangBuiltinIncludesMissing { path }
                if path == toolchain.join("lib/clang/20/include")
        ));
        std::fs::remove_dir_all(toolchain).expect("temporary toolchain must be removed");
    }

    #[test]
    fn clang_builtin_include_resolution_returns_the_version_include_directory() {
        let toolchain = temporary_directory("clang-with-includes");
        let include = toolchain.join("lib/clang/20/include");
        std::fs::create_dir_all(&include).expect("Clang include directory must be created");

        let resolved =
            find_clang_builtin_includes(toolchain.to_str().expect("UTF-8 temporary path"))
                .expect("built-in includes must resolve");

        assert_eq!(PathBuf::from(resolved), include);
        std::fs::remove_dir_all(toolchain).expect("temporary toolchain must be removed");
    }
}
