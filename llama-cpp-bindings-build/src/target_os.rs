use crate::apple_variant::AppleVariant;
use crate::windows_variant::WindowsVariant;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetOs {
    Windows(WindowsVariant),
    Apple(AppleVariant),
    Linux,
    Android,
}

impl TargetOs {
    #[must_use]
    pub fn from_cargo_cfg(cargo_cfg_target_os: &str, cargo_cfg_target_env: &str) -> Option<Self> {
        match cargo_cfg_target_os {
            "windows" => Some(Self::Windows(if cargo_cfg_target_env == "msvc" {
                WindowsVariant::Msvc
            } else {
                WindowsVariant::Other
            })),
            "macos" => Some(Self::Apple(AppleVariant::MacOS)),
            "ios" | "tvos" | "watchos" | "visionos" => Some(Self::Apple(AppleVariant::Other)),
            "android" => Some(Self::Android),
            "linux" => Some(Self::Linux),
            _ => None,
        }
    }

    #[must_use]
    pub const fn is_android(self) -> bool {
        matches!(self, Self::Android)
    }

    #[must_use]
    pub const fn is_msvc(self) -> bool {
        matches!(self, Self::Windows(WindowsVariant::Msvc))
    }
}

#[cfg(test)]
mod tests {
    use super::TargetOs;
    use crate::apple_variant::AppleVariant;
    use crate::windows_variant::WindowsVariant;

    #[test]
    fn windows_is_split_by_its_target_environment() {
        assert_eq!(
            TargetOs::from_cargo_cfg("windows", "msvc"),
            Some(TargetOs::Windows(WindowsVariant::Msvc))
        );
        assert_eq!(
            TargetOs::from_cargo_cfg("windows", "gnu"),
            Some(TargetOs::Windows(WindowsVariant::Other))
        );
    }

    #[test]
    fn macos_is_distinguished_from_the_other_apple_platforms() {
        assert_eq!(
            TargetOs::from_cargo_cfg("macos", ""),
            Some(TargetOs::Apple(AppleVariant::MacOS))
        );

        for apple_os in ["ios", "tvos", "watchos", "visionos"] {
            assert_eq!(
                TargetOs::from_cargo_cfg(apple_os, ""),
                Some(TargetOs::Apple(AppleVariant::Other)),
                "{apple_os} must classify as a non-macOS Apple target"
            );
        }
    }

    #[test]
    fn android_is_not_mistaken_for_linux() {
        assert_eq!(
            TargetOs::from_cargo_cfg("android", ""),
            Some(TargetOs::Android)
        );
        assert_eq!(
            TargetOs::from_cargo_cfg("linux", "gnu"),
            Some(TargetOs::Linux)
        );
    }

    #[test]
    fn an_unsupported_target_os_is_rejected() {
        assert_eq!(TargetOs::from_cargo_cfg("freebsd", ""), None);
    }
}
