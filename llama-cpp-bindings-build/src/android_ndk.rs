use std::env::VarError;
use std::fs;
use std::fs::DirEntry;
use std::io;
use std::path::Path;
use std::path::PathBuf;

use crate::android_architecture::AndroidArchitecture;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::host_os::HostOs;

const DEFAULT_ANDROID_API_LEVEL: &str = "28";

fn detect_ndk_path(
    environment: &BuildEnvironment,
    target_triple: &str,
) -> Result<String, BuildError> {
    environment
        .var("ANDROID_NDK_HOME")
        .map_err(|source| BuildError::AndroidNdkRootNotConfigured {
            target_triple: target_triple.to_owned(),
            source,
        })
}

fn validate_ndk_installation(ndk_path: &str) -> Result<(), BuildError> {
    let ndk_path = Path::new(ndk_path);

    if !ndk_path.exists() {
        return Err(BuildError::AndroidNdkRootMissing {
            path: ndk_path.to_path_buf(),
        });
    }

    let toolchain_file = ndk_path.join("build/cmake/android.toolchain.cmake");

    if !toolchain_file.exists() {
        return Err(BuildError::AndroidNdkToolchainFileMissing {
            path: toolchain_file,
        });
    }

    Ok(())
}

fn detect_api_level(environment: &BuildEnvironment) -> Result<String, BuildError> {
    match environment.var("ANDROID_PLATFORM") {
        Ok(platform) => Ok(platform
            .strip_prefix("android-")
            .unwrap_or(&platform)
            .to_owned()),
        Err(VarError::NotPresent) => Ok(DEFAULT_ANDROID_API_LEVEL.to_owned()),
        Err(source) => Err(BuildError::AndroidPlatformUnreadable { source }),
    }
}

/// The NDK toolchain keeps the built-in headers of its clang in the single version directory
/// under `lib/clang`.
fn find_clang_builtin_includes(toolchain_path: &str) -> Result<String, BuildError> {
    let clang_lib_path = PathBuf::from(toolchain_path).join("lib/clang");
    let mut version_dirs = fs::read_dir(&clang_lib_path)
        .and_then(Iterator::collect::<Result<Vec<DirEntry>, io::Error>>)
        .map_err(|source| BuildError::AndroidNdkClangDirectoryUnreadable {
            path: clang_lib_path.clone(),
            source,
        })?
        .into_iter()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<PathBuf>>();

    version_dirs.sort();

    let [version_dir] = <[PathBuf; 1]>::try_from(version_dirs).map_err(|candidates| {
        if candidates.is_empty() {
            BuildError::AndroidNdkClangVersionDirectoryMissing {
                path: clang_lib_path.clone(),
            }
        } else {
            BuildError::AndroidNdkClangVersionDirectoryAmbiguous {
                path: clang_lib_path.clone(),
                candidates,
            }
        }
    })?;
    let include_path = version_dir.join("include");

    if !include_path.is_dir() {
        return Err(BuildError::AndroidNdkClangBuiltinIncludesMissing { path: include_path });
    }

    Ok(include_path.to_string_lossy().into_owned())
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AndroidNdk {
    pub ndk_path: String,
    pub api_level: String,
    pub abi: &'static str,
    pub sysroot: String,
    pub target_prefix: &'static str,
    pub clang_builtin_includes: String,
}

impl AndroidNdk {
    /// # Errors
    ///
    /// Returns the [`BuildError`] describing why the NDK installation cannot be located, an
    /// environment variable is missing or malformed, the target architecture is unsupported, or
    /// the host platform is not supported by the NDK.
    pub fn detect(
        environment: &BuildEnvironment,
        target_triple: &str,
        cargo_cfg_target_arch: &str,
        host_os: HostOs,
    ) -> Result<Self, BuildError> {
        let ndk_path = detect_ndk_path(environment, target_triple)?;

        validate_ndk_installation(&ndk_path)?;

        let architecture = AndroidArchitecture::from_cargo_cfg(cargo_cfg_target_arch)?;
        let api_level = detect_api_level(environment)?;
        let host_tag = host_os.android_ndk_tag()?;
        let toolchain_path = format!("{ndk_path}/toolchains/llvm/prebuilt/{host_tag}");

        if !Path::new(&toolchain_path).exists() {
            return Err(BuildError::AndroidNdkToolchainDirectoryMissing {
                path: PathBuf::from(toolchain_path),
            });
        }

        let clang_builtin_includes = find_clang_builtin_includes(&toolchain_path)?;

        Ok(Self {
            ndk_path,
            api_level,
            abi: architecture.abi(),
            sysroot: format!("{toolchain_path}/sysroot"),
            target_prefix: architecture.ndk_prefix(),
            clang_builtin_includes,
        })
    }

    #[must_use]
    pub fn android_platform(&self) -> String {
        format!("android-{}", self.api_level)
    }

    #[must_use]
    pub fn cmake_toolchain_file(&self) -> String {
        format!("{}/build/cmake/android.toolchain.cmake", self.ndk_path)
    }

    /// The clang arguments that parse headers the way the NDK compiles them for `target_triple`.
    #[must_use]
    pub fn clang_args(&self, target_triple: &str) -> Vec<String> {
        vec![
            format!("--sysroot={}", self.sysroot),
            format!("-D__ANDROID_API__={}", self.api_level),
            "-D__ANDROID__".to_owned(),
            "-isystem".to_owned(),
            self.clang_builtin_includes.clone(),
            "-isystem".to_owned(),
            format!("{}/usr/include/{}", self.sysroot, self.target_prefix),
            "-isystem".to_owned(),
            format!("{}/usr/include", self.sysroot),
            "-include".to_owned(),
            "stdbool.h".to_owned(),
            "-include".to_owned(),
            "stdint.h".to_owned(),
            format!("--target={target_triple}"),
        ]
    }
}

#[cfg(test)]
pub mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::Path;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use enumflags2::BitFlags;

    use super::AndroidNdk;
    use super::find_clang_builtin_includes;
    use super::validate_ndk_installation;
    use crate::build_environment::BuildEnvironment;
    use crate::build_environment::tests::environment_of;
    use crate::host_os::HostOs;

    pub const FIXTURE_ANDROID_NDK: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/android_ndk");

    const FIXTURE_TOOLCHAIN: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/fixtures/android_ndk/toolchains/llvm/prebuilt/linux-x86_64"
    );

    static NEXT_DIRECTORY_ID: AtomicUsize = AtomicUsize::new(0);

    fn temporary_directory(name: &str) -> PathBuf {
        let id = NEXT_DIRECTORY_ID.fetch_add(1, Ordering::Relaxed);

        std::env::temp_dir().join(format!(
            "llama-cpp-bindings-{name}-{}-{id}",
            std::process::id()
        ))
    }

    fn ndk_at(root: &Path) -> BuildEnvironment {
        environment_of(
            BitFlags::empty(),
            [("ANDROID_NDK_HOME", OsString::from(root))],
        )
    }

    fn detection_failure(
        environment: &BuildEnvironment,
        cargo_cfg_target_arch: &str,
        host_os: HostOs,
    ) -> Option<String> {
        AndroidNdk::detect(
            environment,
            "aarch64-linux-android",
            cargo_cfg_target_arch,
            host_os,
        )
        .err()
        .map(|error| error.to_string())
    }

    #[test]
    fn an_installed_ndk_is_detected_with_the_platform_api_level() {
        let detected = AndroidNdk::detect(
            &environment_of(
                BitFlags::empty(),
                [
                    ("ANDROID_NDK_HOME", OsString::from(FIXTURE_ANDROID_NDK)),
                    ("ANDROID_PLATFORM", OsString::from("android-30")),
                ],
            ),
            "aarch64-linux-android",
            "aarch64",
            HostOs::Linux,
        )
        .unwrap();

        assert_eq!(
            [detected.android_platform(), detected.cmake_toolchain_file(),],
            [
                "android-30".to_owned(),
                format!("{FIXTURE_ANDROID_NDK}/build/cmake/android.toolchain.cmake"),
            ]
        );
        assert_eq!(
            detected,
            AndroidNdk {
                ndk_path: FIXTURE_ANDROID_NDK.to_owned(),
                api_level: "30".to_owned(),
                abi: "arm64-v8a",
                sysroot: format!("{FIXTURE_TOOLCHAIN}/sysroot"),
                target_prefix: "aarch64-linux-android",
                clang_builtin_includes: format!("{FIXTURE_TOOLCHAIN}/lib/clang/20/include"),
            }
        );
    }

    #[test]
    fn the_api_level_is_read_with_or_without_the_android_prefix_and_defaults_to_28() {
        let api_level_of = |environment: &BuildEnvironment| {
            AndroidNdk::detect(
                environment,
                "aarch64-linux-android",
                "aarch64",
                HostOs::Linux,
            )
            .map(|ndk| ndk.api_level)
            .ok()
        };

        assert_eq!(
            [
                api_level_of(&environment_of(
                    BitFlags::empty(),
                    [
                        ("ANDROID_NDK_HOME", OsString::from(FIXTURE_ANDROID_NDK)),
                        ("ANDROID_PLATFORM", OsString::from("33")),
                    ]
                )),
                api_level_of(&ndk_at(Path::new(FIXTURE_ANDROID_NDK))),
            ],
            [Some("33".to_owned()), Some("28".to_owned())]
        );
    }

    #[test]
    fn detection_reports_every_way_the_ndk_cannot_be_used() {
        let fixture_ndk = Path::new(FIXTURE_ANDROID_NDK);
        let missing_root = temporary_directory("missing-ndk-root");

        assert_eq!(
            [
                detection_failure(
                    &environment_of(BitFlags::empty(), []),
                    "aarch64",
                    HostOs::Linux
                ),
                detection_failure(&ndk_at(&missing_root), "aarch64", HostOs::Linux),
                detection_failure(&ndk_at(fixture_ndk), "riscv64", HostOs::Linux),
                detection_failure(
                    &environment_of(
                        BitFlags::empty(),
                        [
                            ("ANDROID_NDK_HOME", OsString::from(FIXTURE_ANDROID_NDK)),
                            ("ANDROID_PLATFORM", OsString::from_vec(vec![0xff])),
                        ]
                    ),
                    "aarch64",
                    HostOs::Linux
                ),
                detection_failure(&ndk_at(fixture_ndk), "aarch64", HostOs::Unsupported),
                detection_failure(&ndk_at(fixture_ndk), "aarch64", HostOs::MacOs),
            ],
            [
                Some(
                    "Android NDK not found for target aarch64-linux-android. Set ANDROID_NDK_HOME."
                        .to_owned()
                ),
                Some(format!(
                    "Android NDK path does not exist: {}",
                    missing_root.display()
                )),
                Some("Unsupported Android target architecture: riscv64".to_owned()),
                Some(
                    "ANDROID_PLATFORM is set but could not be read: environment variable was not \
                     valid unicode: \"\\xFF\""
                        .to_owned()
                ),
                Some("Unsupported host platform for Android NDK".to_owned()),
                Some(format!(
                    "Android NDK toolchain not found at: \
                     {FIXTURE_ANDROID_NDK}/toolchains/llvm/prebuilt/darwin-x86_64"
                )),
            ]
        );
    }

    #[test]
    fn ndk_validation_rejects_a_root_without_a_cmake_toolchain() {
        let root = temporary_directory("ndk-without-toolchain");
        std::fs::create_dir_all(&root).unwrap();

        assert_eq!(
            validate_ndk_installation(root.to_str().unwrap())
                .err()
                .map(|error| error.to_string()),
            Some(format!(
                "Android NDK toolchain file not found: {}",
                root.join("build/cmake/android.toolchain.cmake").display()
            ))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn detection_rejects_a_toolchain_without_a_clang_directory() {
        let root = temporary_directory("ndk-without-clang");
        let toolchain = root.join("toolchains/llvm/prebuilt/linux-x86_64");
        std::fs::create_dir_all(root.join("build/cmake")).unwrap();
        std::fs::write(root.join("build/cmake/android.toolchain.cmake"), "").unwrap();
        std::fs::create_dir_all(&toolchain).unwrap();

        assert_eq!(
            detection_failure(&ndk_at(&root), "aarch64", HostOs::Linux),
            Some(format!(
                "Android NDK Clang directory could not be listed at {}: No such file or directory \
                 (os error 2)",
                toolchain.join("lib/clang").display()
            ))
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clang_builtin_include_resolution_requires_a_version_directory() {
        let toolchain = temporary_directory("clang-without-version");
        std::fs::create_dir_all(toolchain.join("lib/clang")).unwrap();
        std::fs::write(toolchain.join("lib/clang/21"), "").unwrap();

        assert_eq!(
            find_clang_builtin_includes(toolchain.to_str().unwrap())
                .err()
                .map(|error| error.to_string()),
            Some(format!(
                "Android NDK Clang version directory was not found at {}",
                toolchain.join("lib/clang").display()
            ))
        );
        std::fs::remove_dir_all(toolchain).unwrap();
    }

    #[test]
    fn clang_builtin_include_resolution_refuses_to_choose_between_version_directories() {
        let toolchain = temporary_directory("clang-with-two-versions");
        std::fs::create_dir_all(toolchain.join("lib/clang/18/include")).unwrap();
        std::fs::create_dir_all(toolchain.join("lib/clang/17/include")).unwrap();

        assert_eq!(
            find_clang_builtin_includes(toolchain.to_str().unwrap())
                .err()
                .map(|error| error.to_string()),
            Some(format!(
                "Android NDK Clang directory {} holds more than one version directory: {:?}",
                toolchain.join("lib/clang").display(),
                [
                    toolchain.join("lib/clang/17"),
                    toolchain.join("lib/clang/18")
                ]
            ))
        );
        std::fs::remove_dir_all(toolchain).unwrap();
    }

    #[test]
    fn clang_builtin_include_resolution_requires_an_include_directory() {
        let toolchain = temporary_directory("clang-without-includes");
        std::fs::create_dir_all(toolchain.join("lib/clang/20")).unwrap();

        assert_eq!(
            find_clang_builtin_includes(toolchain.to_str().unwrap())
                .err()
                .map(|error| error.to_string()),
            Some(format!(
                "Android NDK Clang built-in include directory was not found at {}",
                toolchain.join("lib/clang/20/include").display()
            ))
        );
        std::fs::remove_dir_all(toolchain).unwrap();
    }
}
