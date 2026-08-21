use std::path::Path;

use crate::glob_paths;
use crate::glob_paths::GlobPathsError;
use crate::host_platform::HostPlatform;

fn extract_single_lib_name(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;

    if let Some(stripped) = stem.strip_prefix("lib") {
        return Some(stripped.to_string());
    }

    if path.extension() == Some(std::ffi::OsStr::new("a")) {
        let renamed_path = path.with_file_name(format!("lib{stem}.a"));

        if let Err(error) = std::fs::rename(path, &renamed_path) {
            println!(
                "cargo:warning=failed to rename {} to {}: {error}",
                path.display(),
                renamed_path.display()
            );
        }
    }

    Some(stem.to_string())
}

pub fn extract_lib_names(cmake_dir: &Path, build_shared_libs: bool) -> Vec<String> {
    extract_lib_names_for(cmake_dir, build_shared_libs, HostPlatform::current())
}

fn extract_lib_names_for(
    cmake_dir: &Path,
    build_shared_libs: bool,
    platform: HostPlatform,
) -> Vec<String> {
    let file_pattern = format!("lib*/{}", platform.link_library_pattern(build_shared_libs));

    let paths = match glob_paths::collect_paths(cmake_dir, &file_pattern) {
        Ok(paths) => paths,
        Err(GlobPathsError::NoMatches { .. }) => Vec::new(),
        Err(error) => panic!("library discovery failed: {error}"),
    };

    paths
        .iter()
        .filter_map(|path| extract_single_lib_name(path))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use crate::scratch_dir::ScratchDir;

    use crate::host_platform::HostPlatform;

    use super::extract_lib_names;
    use super::extract_lib_names_for;
    use super::extract_single_lib_name;

    fn static_or_shared_name(stem: &str, build_shared_libs: bool) -> String {
        HostPlatform::current()
            .link_library_pattern(build_shared_libs)
            .replace('*', stem)
    }

    fn cmake_dir_with(scratch: &ScratchDir, file_name: &str) -> PathBuf {
        let cmake_dir = scratch.path().join("cmake");
        let libs_dir = cmake_dir.join("lib");
        std::fs::create_dir_all(&libs_dir).expect("lib dir must be creatable");
        std::fs::write(libs_dir.join(file_name), b"x").expect("library must be writable");

        cmake_dir
    }

    #[test]
    fn a_lib_prefix_is_stripped_from_the_name() {
        let scratch = ScratchDir::new("libname-prefixed");
        let cmake_dir = cmake_dir_with(&scratch, &static_or_shared_name("libggml", false));

        assert_eq!(
            extract_lib_names(&cmake_dir, false),
            vec!["ggml".to_owned()]
        );
    }

    #[test]
    fn shared_libraries_are_matched_when_requested() {
        let scratch = ScratchDir::new("libname-shared");
        let cmake_dir = cmake_dir_with(&scratch, &static_or_shared_name("libggml", true));

        assert_eq!(extract_lib_names(&cmake_dir, true), vec!["ggml".to_owned()]);
    }

    #[test]
    fn an_unprefixed_archive_is_renamed_and_still_reported() {
        let scratch = ScratchDir::new("libname-rename");
        let cmake_dir = cmake_dir_with(&scratch, "ggml.a");

        let names = extract_lib_names_for(&cmake_dir, false, HostPlatform::Unixlike);

        assert_eq!(names, vec!["ggml".to_owned()]);
        assert!(
            cmake_dir.join("lib").join("libggml.a").exists(),
            "the archive must be renamed so the linker can find it"
        );
    }

    #[test]
    fn a_path_without_a_file_stem_yields_no_name() {
        assert_eq!(extract_single_lib_name(Path::new("..")), None);
    }

    #[test]
    fn a_missing_cmake_directory_yields_no_names() {
        let scratch = ScratchDir::new("libname-absent");

        assert!(extract_lib_names(scratch.path(), false).is_empty());
    }

    #[test]
    fn a_directory_without_libraries_yields_no_names() {
        let scratch = ScratchDir::new("libname-empty");
        let cmake_dir = scratch.path().join("cmake-out");
        std::fs::create_dir_all(&cmake_dir).expect("directory must be creatable");

        assert!(extract_lib_names(&cmake_dir, false).is_empty());
    }

    #[test]
    fn a_failing_rename_is_reported_and_the_name_is_still_returned() {
        let scratch = ScratchDir::new("libname-rename-fails");
        let libs_dir = scratch.path().join("lib");
        std::fs::create_dir_all(&libs_dir).expect("lib dir must be creatable");
        let archive = libs_dir.join("ggml.a");
        std::fs::write(&archive, b"x").expect("archive must be writable");
        // Renaming onto a non-empty directory cannot succeed.
        std::fs::create_dir_all(libs_dir.join("libggml.a")).expect("blocker must be creatable");
        std::fs::write(libs_dir.join("libggml.a/occupied"), b"x").expect("blocker must be filled");

        let name = extract_single_lib_name(&archive);

        assert_eq!(name, Some("ggml".to_owned()));
        assert!(archive.exists(), "the archive must be left where it was");
    }

    #[test]
    fn every_platform_and_link_kind_finds_its_libraries() {
        for platform in [
            HostPlatform::Windows,
            HostPlatform::MacOs,
            HostPlatform::Unixlike,
        ] {
            for build_shared_libs in [false, true] {
                let scratch = ScratchDir::new("libname-platform");
                let libs_dir = scratch.path().join("lib");
                std::fs::create_dir_all(&libs_dir).expect("lib dir must be creatable");
                let name = platform
                    .link_library_pattern(build_shared_libs)
                    .replace('*', "libggml");
                std::fs::write(libs_dir.join(&name), b"x").expect("library must be writable");

                let names = extract_lib_names_for(scratch.path(), build_shared_libs, platform);

                assert_eq!(
                    names,
                    vec!["ggml".to_owned()],
                    "{platform:?}/{build_shared_libs}"
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_library_directory_is_reported_and_skipped() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchDir::new("libname-unreadable");
        let cmake_dir = scratch.path().join("cmake");
        let libs_dir = cmake_dir.join("lib");
        std::fs::create_dir_all(libs_dir.join("nested")).expect("dirs must be creatable");
        std::fs::write(libs_dir.join("nested/libggml.a"), b"x").expect("archive must be writable");
        std::fs::set_permissions(&libs_dir, std::fs::Permissions::from_mode(0o000))
            .expect("permissions must be settable");

        let names = extract_lib_names_for(&cmake_dir, false, HostPlatform::Unixlike);

        std::fs::set_permissions(&libs_dir, std::fs::Permissions::from_mode(0o755))
            .expect("permissions must be restorable for cleanup");

        assert!(
            names.is_empty(),
            "unreadable entries are skipped: {names:?}"
        );
    }

    #[test]
    fn an_unprefixed_non_archive_is_reported_without_renaming() {
        let name = extract_single_lib_name(Path::new("/build/lib/ggml.dylib"));

        assert_eq!(
            name,
            Some("ggml".to_owned()),
            "only .a archives are renamed; other kinds are reported as-is"
        );
    }
}
