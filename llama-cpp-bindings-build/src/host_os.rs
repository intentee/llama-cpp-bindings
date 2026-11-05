use std::env::consts::OS;

use crate::build_error::BuildError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostOs {
    Linux,
    MacOs,
    Unsupported,
}

impl HostOs {
    /// The operating system the build script runs on, named by [`std::env::consts::OS`].
    #[must_use]
    pub fn current() -> Self {
        Self::named(OS)
    }

    fn named(operating_system: &str) -> Self {
        match operating_system {
            "linux" => Self::Linux,
            "macos" => Self::MacOs,
            _ => Self::Unsupported,
        }
    }

    /// The directory name of the prebuilt Android NDK toolchain for this host.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::AndroidNdkUnsupportedHostPlatform`] for a host the NDK ships no
    /// toolchain for.
    pub const fn android_ndk_tag(self) -> Result<&'static str, BuildError> {
        match self {
            Self::Linux => Ok("linux-x86_64"),
            Self::MacOs => Ok("darwin-x86_64"),
            Self::Unsupported => Err(BuildError::AndroidNdkUnsupportedHostPlatform),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HostOs;

    #[test]
    fn hosts_are_named_like_the_standard_library_names_them() {
        assert_eq!(
            ["linux", "macos", "windows"].map(HostOs::named),
            [HostOs::Linux, HostOs::MacOs, HostOs::Unsupported]
        );
    }

    #[test]
    fn every_supported_host_names_its_prebuilt_ndk_toolchain() {
        assert_eq!(
            [HostOs::Linux, HostOs::MacOs].map(|host_os| host_os.android_ndk_tag().ok()),
            [Some("linux-x86_64"), Some("darwin-x86_64")]
        );
    }

    #[test]
    fn an_unsupported_host_has_no_ndk_toolchain() {
        assert_eq!(
            HostOs::Unsupported
                .android_ndk_tag()
                .err()
                .map(|error| error.to_string()),
            Some("Unsupported host platform for Android NDK".to_owned())
        );
    }
}
