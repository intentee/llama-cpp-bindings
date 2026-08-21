use std::path::Path;

use walkdir::DirEntry;

use crate::glob_paths;

const WRAPPER_TRACKING_PATTERNS: &[&str] = &["wrapper*.h", "wrapper_*.cpp"];

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with('.'))
}

fn is_cmake_file(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .is_some_and(|name| name.starts_with("CMake"))
}

pub fn register_rebuild_triggers(wrapper_dir: &Path, llama_src: &Path) {
    println!("cargo:rerun-if-changed=build.rs");

    for pattern in WRAPPER_TRACKING_PATTERNS {
        match glob_paths::collect_paths(wrapper_dir, pattern) {
            Ok(paths) => {
                for path in paths {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
            Err(error) => panic!("wrapper rebuild tracking failed: {error}"),
        }
    }

    println!("cargo:rerun-if-env-changed=LLAMA_LIB_PROFILE");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_SHARED_LIBS");
    println!("cargo:rerun-if-env-changed=LLAMA_STATIC_CRT");
    println!("cargo:rerun-if-env-changed=LLAMA_CMAKE_BUILD_DIR_OVERRIDE");

    let source_directories = [
        llama_src.join("src"),
        llama_src.join("ggml/src"),
        llama_src.join("common"),
    ];

    for entry in walkdir::WalkDir::new(llama_src)
        .into_iter()
        .filter_entry(|entry| !is_hidden(entry))
    {
        let Ok(entry) = entry else {
            continue;
        };

        let is_source_child = source_directories
            .iter()
            .any(|source_dir| entry.path().starts_with(source_dir));

        if is_cmake_file(&entry) || is_source_child {
            println!("cargo:rerun-if-changed={}", entry.path().display());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use crate::scratch_dir::ScratchDir;

    use super::is_cmake_file;
    use super::is_hidden;
    use super::register_rebuild_triggers;

    fn entry_for(path: &Path) -> walkdir::DirEntry {
        walkdir::WalkDir::new(path)
            .min_depth(1)
            .max_depth(1)
            .into_iter()
            .next()
            .expect("the directory must contain one entry")
            .expect("the entry must be readable")
    }

    fn wrapper_dir_with_sources(scratch: &ScratchDir) -> PathBuf {
        let wrapper_dir = scratch.path().join("sys");
        std::fs::create_dir_all(&wrapper_dir).expect("wrapper dir must be creatable");
        std::fs::write(wrapper_dir.join("wrapper.h"), b"x").expect("header must be writable");
        std::fs::write(wrapper_dir.join("wrapper_reasoning.cpp"), b"x")
            .expect("source must be writable");

        wrapper_dir
    }

    fn llama_src_tree(scratch: &ScratchDir) -> PathBuf {
        let llama_src = scratch.path().join("llama.cpp");
        std::fs::create_dir_all(llama_src.join("src")).expect("src must be creatable");
        std::fs::create_dir_all(llama_src.join("ggml/src")).expect("ggml/src must be creatable");
        std::fs::create_dir_all(llama_src.join("common")).expect("common must be creatable");
        std::fs::create_dir_all(llama_src.join(".git")).expect(".git must be creatable");
        std::fs::write(llama_src.join("CMakeLists.txt"), b"x").expect("cmake must be writable");
        std::fs::write(llama_src.join("src/llama.cpp"), b"x").expect("source must be writable");
        std::fs::write(llama_src.join(".git/config"), b"x").expect("git file must be writable");

        llama_src
    }

    #[test]
    fn dot_prefixed_entries_are_hidden() {
        let scratch = ScratchDir::new("hidden-yes");
        std::fs::create_dir_all(scratch.path().join(".git")).expect("dir must be creatable");

        assert!(is_hidden(&entry_for(scratch.path())));
    }

    #[test]
    fn ordinary_entries_are_not_hidden() {
        let scratch = ScratchDir::new("hidden-no");
        std::fs::write(scratch.path().join("visible.txt"), b"x").expect("file must be writable");

        assert!(!is_hidden(&entry_for(scratch.path())));
    }

    #[test]
    fn cmake_prefixed_entries_are_recognised() {
        let scratch = ScratchDir::new("cmake-yes");
        std::fs::write(scratch.path().join("CMakeLists.txt"), b"x").expect("file must be writable");

        assert!(is_cmake_file(&entry_for(scratch.path())));
    }

    #[test]
    fn other_entries_are_not_cmake_files() {
        let scratch = ScratchDir::new("cmake-no");
        std::fs::write(scratch.path().join("readme.md"), b"x").expect("file must be writable");

        assert!(!is_cmake_file(&entry_for(scratch.path())));
    }

    #[test]
    fn registering_triggers_walks_wrappers_and_sources() {
        let scratch = ScratchDir::new("triggers-ok");
        let wrapper_dir = wrapper_dir_with_sources(&scratch);
        let llama_src = llama_src_tree(&scratch);

        register_rebuild_triggers(&wrapper_dir, &llama_src);
    }

    #[test]
    #[should_panic(expected = "wrapper rebuild tracking failed")]
    fn a_wrapper_directory_without_sources_panics() {
        let scratch = ScratchDir::new("triggers-missing");
        let wrapper_dir = scratch.path().join("empty");
        std::fs::create_dir_all(&wrapper_dir).expect("wrapper dir must be creatable");

        register_rebuild_triggers(&wrapper_dir, &llama_src_tree(&scratch));
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_directories_are_skipped_during_the_walk() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchDir::new("triggers-unreadable");
        let wrapper_dir = wrapper_dir_with_sources(&scratch);
        let llama_src = llama_src_tree(&scratch);
        let locked = llama_src.join("src/locked");
        std::fs::create_dir_all(&locked).expect("nested dir must be creatable");
        std::fs::write(locked.join("inner.c"), b"x").expect("file must be writable");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("permissions must be settable");

        register_rebuild_triggers(&wrapper_dir, &llama_src);

        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))
            .expect("permissions must be restorable for cleanup");
    }
}
