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
