/// Resolves from an OS and architecture name rather than a `cfg!` branch, so
/// every platform's triple stays exercisable from any host.
fn target_triple_for(operating_system: &str, arch: &str) -> String {
    match operating_system {
        "windows" => format!("{arch}-pc-windows-msvc"),
        "macos" => format!("{arch}-apple-darwin"),
        "linux" => format!("{arch}-unknown-linux-gnu"),
        unsupported => panic!("build tests do not support host operating system {unsupported}"),
    }
}

pub fn host_target_triple() -> String {
    target_triple_for(std::env::consts::OS, std::env::consts::ARCH)
}

#[cfg(test)]
mod tests {
    use super::host_target_triple;
    use super::target_triple_for;

    #[test]
    fn every_supported_host_maps_to_its_triple() {
        assert_eq!(
            target_triple_for("windows", "x86_64"),
            "x86_64-pc-windows-msvc"
        );
        assert_eq!(
            target_triple_for("macos", "aarch64"),
            "aarch64-apple-darwin"
        );
        assert_eq!(
            target_triple_for("linux", "x86_64"),
            "x86_64-unknown-linux-gnu"
        );
    }

    #[test]
    #[should_panic(expected = "do not support host operating system")]
    fn an_unsupported_host_is_rejected() {
        let _ = target_triple_for("plan9", "x86_64");
    }

    #[test]
    fn the_host_triple_names_the_running_platform() {
        assert_eq!(
            host_target_triple(),
            target_triple_for(std::env::consts::OS, std::env::consts::ARCH)
        );
    }
}
