use crate::build_error::BuildError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AndroidArchitecture {
    Aarch64,
    Armv7,
    X86_64,
    X86,
}

impl AndroidArchitecture {
    /// # Errors
    ///
    /// Returns [`BuildError::UnsupportedAndroidTarget`] for an architecture the NDK does not
    /// build for.
    pub fn from_cargo_cfg(cargo_cfg_target_arch: &str) -> Result<Self, BuildError> {
        match cargo_cfg_target_arch {
            "aarch64" => Ok(Self::Aarch64),
            "arm" => Ok(Self::Armv7),
            "x86_64" => Ok(Self::X86_64),
            "x86" => Ok(Self::X86),
            unsupported => Err(BuildError::UnsupportedAndroidTarget {
                cargo_cfg_target_arch: unsupported.to_owned(),
            }),
        }
    }

    #[must_use]
    pub const fn abi(self) -> &'static str {
        match self {
            Self::Aarch64 => "arm64-v8a",
            Self::Armv7 => "armeabi-v7a",
            Self::X86_64 => "x86_64",
            Self::X86 => "x86",
        }
    }

    #[must_use]
    pub const fn ndk_prefix(self) -> &'static str {
        match self {
            Self::Aarch64 => "aarch64-linux-android",
            Self::Armv7 => "arm-linux-androideabi",
            Self::X86_64 => "x86_64-linux-android",
            Self::X86 => "i686-linux-android",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AndroidArchitecture;

    #[test]
    fn every_supported_android_architecture_maps_to_its_abi_and_ndk_prefix() {
        assert_eq!(
            ["aarch64", "arm", "x86_64", "x86"].map(|cargo_cfg_target_arch| {
                AndroidArchitecture::from_cargo_cfg(cargo_cfg_target_arch)
                    .map(|architecture| [architecture.abi(), architecture.ndk_prefix()])
                    .ok()
            }),
            [
                Some(["arm64-v8a", "aarch64-linux-android"]),
                Some(["armeabi-v7a", "arm-linux-androideabi"]),
                Some(["x86_64", "x86_64-linux-android"]),
                Some(["x86", "i686-linux-android"]),
            ]
        );
    }
}
