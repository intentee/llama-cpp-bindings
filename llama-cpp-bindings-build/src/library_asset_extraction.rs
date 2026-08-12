use std::path::{Path, PathBuf};

use glob::glob;

use crate::debug_log;
use crate::host_platform::HostPlatform;

pub fn extract_lib_assets(cmake_dir: &Path) -> Vec<PathBuf> {
    extract_lib_assets_for(cmake_dir, HostPlatform::current())
}

fn extract_lib_assets_for(cmake_dir: &Path, platform: HostPlatform) -> Vec<PathBuf> {
    let libs_dir = cmake_dir.join(platform.shared_library_dir());
    let pattern = libs_dir.join(platform.shared_library_pattern());
    debug_log!("Extract lib assets {}", pattern.display());

    let pattern_str = pattern.to_string_lossy();
    let mut files = Vec::new();

    let Ok(entries) = glob(&pattern_str) else {
        println!("cargo:warning=failed to glob shared lib pattern: {pattern_str}");

        return files;
    };

    for entry in entries {
        match entry {
            Ok(path) => files.push(path),
            Err(error) => eprintln!("cargo:warning=glob error: {error}"),
        }
    }

    files
}

#[cfg(test)]
mod tests {
    use crate::scratch_dir::ScratchDir;

    use crate::host_platform::HostPlatform;

    use super::extract_lib_assets;
    use super::extract_lib_assets_for;

    fn shared_lib_name(stem: &str) -> String {
        HostPlatform::current()
            .shared_library_pattern()
            .replace('*', stem)
    }

    fn assets_dir_name() -> &'static str {
        HostPlatform::current().shared_library_dir()
    }

    #[test]
    fn shared_libraries_in_the_assets_directory_are_collected() {
        let scratch = ScratchDir::new("assets-present");
        let assets_dir = scratch.path().join(assets_dir_name());
        std::fs::create_dir_all(&assets_dir).expect("assets directory must be creatable");
        std::fs::write(assets_dir.join(shared_lib_name("libggml")), b"x")
            .expect("shared library must be writable");
        std::fs::write(assets_dir.join("notes.txt"), b"x").expect("text file must be writable");

        let found = extract_lib_assets(scratch.path());

        assert_eq!(found.len(), 1, "only the shared library matches: {found:?}");
        assert!(
            found[0].ends_with(shared_lib_name("libggml")),
            "got: {found:?}"
        );
    }

    #[test]
    fn a_missing_assets_directory_yields_no_files() {
        let scratch = ScratchDir::new("assets-absent");

        assert!(extract_lib_assets(scratch.path()).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_assets_directory_is_reported_and_skipped() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchDir::new("assets-unreadable");
        let assets_dir = scratch.path().join(assets_dir_name());
        std::fs::create_dir_all(&assets_dir).expect("assets directory must be creatable");
        std::fs::write(assets_dir.join(shared_lib_name("libggml")), b"x")
            .expect("shared library must be writable");
        std::fs::set_permissions(&assets_dir, std::fs::Permissions::from_mode(0o000))
            .expect("permissions must be settable");

        let found = extract_lib_assets(scratch.path());

        std::fs::set_permissions(&assets_dir, std::fs::Permissions::from_mode(0o755))
            .expect("permissions must be restorable for cleanup");

        assert!(
            found.is_empty(),
            "unreadable entries are skipped: {found:?}"
        );
    }

    #[test]
    fn every_platform_collects_its_own_shared_library_flavour() {
        for platform in [
            HostPlatform::Windows,
            HostPlatform::MacOs,
            HostPlatform::Unixlike,
        ] {
            let scratch = ScratchDir::new("assets-platform");
            let assets_dir = scratch.path().join(platform.shared_library_dir());
            std::fs::create_dir_all(&assets_dir).expect("assets dir must be creatable");
            let name = platform.shared_library_pattern().replace('*', "libggml");
            std::fs::write(assets_dir.join(&name), b"x").expect("library must be writable");

            let found = extract_lib_assets_for(scratch.path(), platform);

            assert_eq!(found.len(), 1, "{platform:?} -> {found:?}");
            assert!(found[0].ends_with(&name));
        }
    }

    #[test]
    fn an_invalid_glob_pattern_is_reported_and_yields_no_files() {
        let scratch = ScratchDir::new("assets-badglob");
        let cmake_dir = scratch.path().join("a**b");
        std::fs::create_dir_all(cmake_dir.join(assets_dir_name()))
            .expect("assets dir must be creatable");

        assert!(extract_lib_assets(&cmake_dir).is_empty());
    }
}
