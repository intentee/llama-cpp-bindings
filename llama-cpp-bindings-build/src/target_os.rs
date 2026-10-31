use crate::android_ndk::AndroidNdk;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::host_os::HostOs;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TargetOs {
    Apple,
    Linux,
    Android(AndroidNdk),
}

impl TargetOs {
    /// The operating system cargo builds for, with the NDK it needs when that is Android.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when `CARGO_CFG_TARGET_OS` cannot be read,
    /// [`BuildError::UnsupportedTargetOs`] for an operating system this crate does not build
    /// for, or the [`BuildError`] of [`AndroidNdk::detect`] when an Android target has no
    /// usable NDK.
    pub fn detect(
        environment: &BuildEnvironment,
        target_triple: &str,
        cargo_cfg_target_arch: &str,
        host_os: HostOs,
    ) -> Result<Self, BuildError> {
        let cargo_cfg_target_os = environment.required("CARGO_CFG_TARGET_OS")?;

        match cargo_cfg_target_os.as_str() {
            "macos" | "ios" | "tvos" | "watchos" | "visionos" => Ok(Self::Apple),
            "android" => Ok(Self::Android(AndroidNdk::detect(
                environment,
                target_triple,
                cargo_cfg_target_arch,
                host_os,
            )?)),
            "linux" => Ok(Self::Linux),
            _ => Err(BuildError::UnsupportedTargetOs {
                cargo_cfg_target_os,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;

    use enumflags2::BitFlags;

    use super::TargetOs;
    use crate::android_ndk::AndroidNdk;
    use crate::android_ndk::tests::FIXTURE_ANDROID_NDK;
    use crate::build_environment::tests::environment_of;
    use crate::host_os::HostOs;

    fn detected(cargo_cfg_target_os: &str) -> Result<TargetOs, String> {
        let environment = environment_of(
            BitFlags::empty(),
            [
                ("CARGO_CFG_TARGET_OS", OsString::from(cargo_cfg_target_os)),
                ("ANDROID_NDK_HOME", OsString::from(FIXTURE_ANDROID_NDK)),
            ],
        );

        TargetOs::detect(
            &environment,
            "aarch64-linux-android",
            "aarch64",
            HostOs::Linux,
        )
        .map_err(|error| error.to_string())
    }

    #[test]
    fn every_apple_platform_is_an_apple_target() {
        assert_eq!(
            ["macos", "ios", "tvos", "watchos", "visionos"].map(detected),
            [(); 5].map(|()| Ok(TargetOs::Apple))
        );
    }

    #[test]
    fn android_is_not_mistaken_for_linux_and_carries_its_ndk() {
        let environment = environment_of(
            BitFlags::empty(),
            [("ANDROID_NDK_HOME", OsString::from(FIXTURE_ANDROID_NDK))],
        );

        assert_eq!(
            [detected("android"), detected("linux")],
            [
                Ok(TargetOs::Android(
                    AndroidNdk::detect(
                        &environment,
                        "aarch64-linux-android",
                        "aarch64",
                        HostOs::Linux
                    )
                    .unwrap()
                )),
                Ok(TargetOs::Linux)
            ]
        );
    }

    #[test]
    fn an_unsupported_or_unusable_target_is_refused() {
        assert_eq!(
            [
                detected("freebsd"),
                TargetOs::detect(
                    &environment_of(
                        BitFlags::empty(),
                        [("CARGO_CFG_TARGET_OS", "android".into())]
                    ),
                    "aarch64-linux-android",
                    "aarch64",
                    HostOs::Linux
                )
                .map_err(|error| error.to_string()),
            ],
            [
                Err("unsupported target operating system: freebsd".to_owned()),
                Err(
                    "Android NDK not found for target aarch64-linux-android. Set ANDROID_NDK_HOME."
                        .to_owned()
                ),
            ]
        );
    }
}
