use std::path::Path;

use crate::debug_log;
use crate::library_asset_extraction::extract_lib_assets;

pub fn copy_shared_libraries(cmake_dir: &Path, target_dir: &Path) {
    let assets = extract_lib_assets(cmake_dir);

    for asset in &assets {
        let filename = asset.file_name().unwrap_or_else(|| asset.as_os_str());

        hard_link_if_missing(asset, &target_dir.join(filename));

        let examples_dir = target_dir.join("examples");

        if examples_dir.exists() {
            hard_link_if_missing(asset, &examples_dir.join(filename));
        }

        let deps_dir = target_dir.join("deps");
        hard_link_if_missing(asset, &deps_dir.join(filename));
    }
}

fn hard_link_if_missing(source: &Path, destination: &Path) {
    if destination.exists() {
        return;
    }

    debug_log!(
        "HARD LINK {} TO {}",
        source.display(),
        destination.display()
    );

    if let Err(error) = std::fs::hard_link(source, destination) {
        println!(
            "cargo:warning=failed to hard link {} to {}: {error}",
            source.display(),
            destination.display()
        );
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use serial_test::serial;

    use crate::scratch_dir::ScratchDir;

    use super::copy_shared_libraries;
    use super::hard_link_if_missing;

    fn shared_lib_name(stem: &str) -> String {
        crate::host_platform::HostPlatform::current()
            .shared_library_pattern()
            .replace('*', stem)
    }

    fn cmake_dir_with_one_library(scratch: &ScratchDir) -> PathBuf {
        let cmake_dir = scratch.path().join("cmake");
        let assets_dir =
            cmake_dir.join(crate::host_platform::HostPlatform::current().shared_library_dir());
        std::fs::create_dir_all(&assets_dir).expect("assets directory must be creatable");
        std::fs::write(assets_dir.join(shared_lib_name("libggml")), b"payload")
            .expect("shared library must be writable");

        cmake_dir
    }

    fn target_dir_with_deps(scratch: &ScratchDir) -> PathBuf {
        let target_dir = scratch.path().join("target");
        std::fs::create_dir_all(target_dir.join("deps")).expect("deps must be creatable");

        target_dir
    }

    #[test]
    fn libraries_are_linked_into_the_target_root_and_deps() {
        let scratch = ScratchDir::new("shared-basic");
        let cmake_dir = cmake_dir_with_one_library(&scratch);
        let target_dir = target_dir_with_deps(&scratch);

        copy_shared_libraries(&cmake_dir, &target_dir);

        let name = shared_lib_name("libggml");
        assert!(target_dir.join(&name).exists(), "root copy must exist");
        assert!(
            target_dir.join("deps").join(&name).exists(),
            "deps copy must exist"
        );
        assert!(
            !target_dir.join("examples").join(&name).exists(),
            "examples must be skipped when the directory is absent"
        );
    }

    #[test]
    fn an_existing_examples_directory_also_receives_the_library() {
        let scratch = ScratchDir::new("shared-examples");
        let cmake_dir = cmake_dir_with_one_library(&scratch);
        let target_dir = target_dir_with_deps(&scratch);
        std::fs::create_dir_all(target_dir.join("examples")).expect("examples must be creatable");

        copy_shared_libraries(&cmake_dir, &target_dir);

        assert!(
            target_dir
                .join("examples")
                .join(shared_lib_name("libggml"))
                .exists()
        );
    }

    #[test]
    fn an_existing_destination_is_left_untouched() {
        let scratch = ScratchDir::new("shared-existing");
        let source = scratch.path().join("source.bin");
        let destination = scratch.path().join("destination.bin");
        std::fs::write(&source, b"fresh").expect("source must be writable");
        std::fs::write(&destination, b"original").expect("destination must be writable");

        hard_link_if_missing(&source, &destination);

        assert_eq!(
            std::fs::read(&destination).expect("destination must be readable"),
            b"original",
            "an existing destination must not be replaced"
        );
    }

    #[test]
    fn a_failing_link_is_reported_without_panicking() {
        let scratch = ScratchDir::new("shared-failure");
        let source = scratch.path().join("source.bin");
        std::fs::write(&source, b"payload").expect("source must be writable");
        let destination = scratch.path().join("absent-parent").join("destination.bin");

        hard_link_if_missing(&source, &destination);

        assert!(
            !destination.exists(),
            "linking into a missing parent cannot succeed"
        );
    }

    #[test]
    fn a_cmake_directory_without_libraries_links_nothing() {
        let scratch = ScratchDir::new("shared-empty");
        let target_dir = target_dir_with_deps(&scratch);

        copy_shared_libraries(Path::new(scratch.path()), &target_dir);

        assert!(
            std::fs::read_dir(target_dir.join("deps"))
                .expect("deps must be readable")
                .next()
                .is_none()
        );
    }

    #[test]
    #[serial]
    fn debug_logging_reports_each_link() {
        let scratch = ScratchDir::new("shared-debug");
        let source = scratch.path().join("source.bin");
        std::fs::write(&source, b"payload").expect("source must be writable");
        unsafe { std::env::set_var("BUILD_DEBUG", "1") };

        hard_link_if_missing(&source, &scratch.path().join("linked.bin"));

        unsafe { std::env::remove_var("BUILD_DEBUG") };

        assert!(scratch.path().join("linked.bin").exists());
    }
}
