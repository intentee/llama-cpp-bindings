/// The platform whose library-naming conventions apply to a build.
///
/// Taking this as a parameter rather than branching on `cfg!` inline keeps the
/// naming rules for every platform exercisable from any host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostPlatform {
    Windows,
    MacOs,
    Unixlike,
}

impl HostPlatform {
    #[must_use]
    pub fn current() -> Self {
        Self::from_os(std::env::consts::OS)
    }

    /// Resolves from an OS name rather than a `cfg!` branch, so every
    /// platform's rules stay reachable from any host.
    #[must_use]
    pub fn from_os(os: &str) -> Self {
        match os {
            "windows" => Self::Windows,
            "macos" => Self::MacOs,
            _ => Self::Unixlike,
        }
    }

    /// Glob for shared libraries produced by a cmake build.
    #[must_use]
    pub const fn shared_library_pattern(self) -> &'static str {
        match self {
            Self::Windows => "*.dll",
            Self::MacOs => "*.dylib",
            Self::Unixlike => "*.so",
        }
    }

    /// Directory a cmake install places shared libraries into.
    #[must_use]
    pub const fn shared_library_dir(self) -> &'static str {
        match self {
            Self::Windows => "bin",
            Self::MacOs | Self::Unixlike => "lib",
        }
    }

    /// Glob for the libraries the linker should be told about.
    #[must_use]
    pub const fn link_library_pattern(self, build_shared_libs: bool) -> &'static str {
        match self {
            Self::Windows => "*.lib",
            Self::MacOs if build_shared_libs => "*.dylib",
            Self::Unixlike if build_shared_libs => "*.so",
            Self::MacOs | Self::Unixlike => "*.a",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HostPlatform;

    #[test]
    fn every_os_name_resolves_to_its_platform() {
        assert_eq!(HostPlatform::from_os("windows"), HostPlatform::Windows);
        assert_eq!(HostPlatform::from_os("macos"), HostPlatform::MacOs);
        assert_eq!(HostPlatform::from_os("linux"), HostPlatform::Unixlike);
        assert_eq!(HostPlatform::from_os("freebsd"), HostPlatform::Unixlike);
    }

    #[test]
    fn the_current_platform_resolves_from_the_running_os() {
        assert_eq!(
            HostPlatform::current(),
            HostPlatform::from_os(std::env::consts::OS)
        );
    }

    #[test]
    fn every_platform_names_its_shared_libraries() {
        assert_eq!(HostPlatform::Windows.shared_library_pattern(), "*.dll");
        assert_eq!(HostPlatform::MacOs.shared_library_pattern(), "*.dylib");
        assert_eq!(HostPlatform::Unixlike.shared_library_pattern(), "*.so");
    }

    #[test]
    fn only_windows_installs_shared_libraries_beside_the_binaries() {
        assert_eq!(HostPlatform::Windows.shared_library_dir(), "bin");
        assert_eq!(HostPlatform::MacOs.shared_library_dir(), "lib");
        assert_eq!(HostPlatform::Unixlike.shared_library_dir(), "lib");
    }

    #[test]
    fn link_patterns_follow_the_platform_and_link_kind() {
        assert_eq!(HostPlatform::Windows.link_library_pattern(false), "*.lib");
        assert_eq!(HostPlatform::Windows.link_library_pattern(true), "*.lib");
        assert_eq!(HostPlatform::MacOs.link_library_pattern(false), "*.a");
        assert_eq!(HostPlatform::MacOs.link_library_pattern(true), "*.dylib");
        assert_eq!(HostPlatform::Unixlike.link_library_pattern(false), "*.a");
        assert_eq!(HostPlatform::Unixlike.link_library_pattern(true), "*.so");
    }
}
