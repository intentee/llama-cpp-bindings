#[derive(Debug, Clone, Copy)]
pub enum WindowsVariant {
    Msvc,
    Other,
}

#[derive(Debug, Clone, Copy)]
pub enum AppleVariant {
    MacOS,
    Other,
}

#[derive(Debug)]
pub enum TargetOs {
    Windows(WindowsVariant),
    Apple(AppleVariant),
    Linux,
    Android,
}

impl TargetOs {
    pub fn from_target_triple(target_triple: &str) -> Result<Self, String> {
        if target_triple.contains("windows") {
            if target_triple.ends_with("-windows-msvc") {
                Ok(TargetOs::Windows(WindowsVariant::Msvc))
            } else {
                Ok(TargetOs::Windows(WindowsVariant::Other))
            }
        } else if target_triple.contains("apple") {
            if target_triple.ends_with("-apple-darwin") {
                Ok(TargetOs::Apple(AppleVariant::MacOS))
            } else {
                Ok(TargetOs::Apple(AppleVariant::Other))
            }
        } else if target_triple.contains("android") {
            Ok(TargetOs::Android)
        } else if target_triple.contains("linux") {
            Ok(TargetOs::Linux)
        } else {
            Err(format!("Unsupported target triple: {target_triple}"))
        }
    }

    pub fn is_android(&self) -> bool {
        matches!(self, TargetOs::Android)
    }

    pub fn is_msvc(&self) -> bool {
        matches!(self, TargetOs::Windows(WindowsVariant::Msvc))
    }
}

#[cfg(test)]
mod tests {
    use super::AppleVariant;
    use super::TargetOs;
    use super::WindowsVariant;

    #[test]
    fn msvc_triple_is_distinguished_from_other_windows_targets() {
        assert!(matches!(
            TargetOs::from_target_triple("x86_64-pc-windows-msvc"),
            Ok(TargetOs::Windows(WindowsVariant::Msvc))
        ));
        assert!(matches!(
            TargetOs::from_target_triple("x86_64-pc-windows-gnu"),
            Ok(TargetOs::Windows(WindowsVariant::Other))
        ));
    }

    #[test]
    fn macos_triple_is_distinguished_from_other_apple_targets() {
        assert!(matches!(
            TargetOs::from_target_triple("aarch64-apple-darwin"),
            Ok(TargetOs::Apple(AppleVariant::MacOS))
        ));
        assert!(matches!(
            TargetOs::from_target_triple("aarch64-apple-ios"),
            Ok(TargetOs::Apple(AppleVariant::Other))
        ));
    }

    #[test]
    fn android_is_matched_before_linux_despite_containing_linux() {
        assert!(matches!(
            TargetOs::from_target_triple("aarch64-linux-android"),
            Ok(TargetOs::Android)
        ));
    }

    #[test]
    fn plain_linux_triple_is_linux() {
        assert!(matches!(
            TargetOs::from_target_triple("x86_64-unknown-linux-gnu"),
            Ok(TargetOs::Linux)
        ));
    }

    #[test]
    fn an_unsupported_triple_reports_the_triple() {
        let error = TargetOs::from_target_triple("sparc-unknown-none")
            .expect_err("unsupported triple must fail");

        assert!(error.contains("sparc-unknown-none"), "got: {error}");
    }

    #[test]
    fn is_android_and_is_msvc_only_match_their_own_variant() {
        let android = TargetOs::from_target_triple("aarch64-linux-android").expect("android");
        let msvc = TargetOs::from_target_triple("x86_64-pc-windows-msvc").expect("msvc");
        let linux = TargetOs::from_target_triple("x86_64-unknown-linux-gnu").expect("linux");

        assert!(android.is_android());
        assert!(!android.is_msvc());
        assert!(msvc.is_msvc());
        assert!(!msvc.is_android());
        assert!(!linux.is_android());
        assert!(!linux.is_msvc());
    }

    #[test]
    fn variants_are_debug_printable_for_build_diagnostics() {
        let msvc = TargetOs::from_target_triple("x86_64-pc-windows-msvc").expect("msvc");
        let macos = TargetOs::from_target_triple("aarch64-apple-darwin").expect("macos");

        assert!(format!("{msvc:?}").contains("Msvc"));
        assert!(format!("{macos:?}").contains("MacOS"));
    }
}
