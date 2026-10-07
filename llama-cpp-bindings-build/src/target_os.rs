use crate::apple_variant::AppleVariant;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetOs {
    Apple(AppleVariant),
    Linux,
    Android,
}

impl TargetOs {
    #[must_use]
    pub fn from_cargo_cfg(cargo_cfg_target_os: &str) -> Option<Self> {
        match cargo_cfg_target_os {
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
}

#[cfg(test)]
mod tests {
    use super::TargetOs;
    use crate::apple_variant::AppleVariant;

    #[test]
    fn macos_is_distinguished_from_the_other_apple_platforms() {
        assert_eq!(
            TargetOs::from_cargo_cfg("macos"),
            Some(TargetOs::Apple(AppleVariant::MacOS))
        );

        for apple_os in ["ios", "tvos", "watchos", "visionos"] {
            assert_eq!(
                TargetOs::from_cargo_cfg(apple_os),
                Some(TargetOs::Apple(AppleVariant::Other)),
                "{apple_os} must classify as a non-macOS Apple target"
            );
        }
    }

    #[test]
    fn android_is_not_mistaken_for_linux() {
        assert_eq!(TargetOs::from_cargo_cfg("android"), Some(TargetOs::Android));
        assert_eq!(TargetOs::from_cargo_cfg("linux"), Some(TargetOs::Linux));
    }

    #[test]
    fn only_android_needs_the_android_stdlib_handling() {
        assert!(TargetOs::Android.is_android());
        assert!(!TargetOs::Linux.is_android());
    }

    #[test]
    fn an_unsupported_target_os_is_rejected() {
        for unsupported_target_os in ["freebsd", "windows"] {
            assert_eq!(TargetOs::from_cargo_cfg(unsupported_target_os), None);
        }
    }
}
