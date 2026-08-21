use std::path::{Path, PathBuf};

use crate::glob_paths;
use crate::glob_paths::GlobPathsError;
use crate::host_platform::HostPlatform;

pub fn extract_lib_assets(cmake_dir: &Path) -> Vec<PathBuf> {
    extract_lib_assets_for(cmake_dir, HostPlatform::current())
}

fn extract_lib_assets_for(cmake_dir: &Path, platform: HostPlatform) -> Vec<PathBuf> {
    let libs_dir = cmake_dir.join(platform.shared_library_dir());

    match glob_paths::collect_paths(&libs_dir, platform.shared_library_pattern()) {
        Ok(paths) => paths,
        Err(GlobPathsError::NoMatches { .. }) => Vec::new(),
        Err(error) => panic!("shared library discovery failed: {error}"),
    }
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
    fn a_directory_holding_glob_metacharacters_still_yields_its_libraries() {
        let scratch = ScratchDir::new("assets-metachars");
        let cmake_dir = scratch.path().join("a[b]c");
        let assets_dir = cmake_dir.join(assets_dir_name());
        std::fs::create_dir_all(&assets_dir).expect("assets dir must be creatable");
        std::fs::write(assets_dir.join(shared_lib_name("probe")), b"x")
            .expect("library must be writable");

        assert_eq!(extract_lib_assets(&cmake_dir).len(), 1);
    }
}
