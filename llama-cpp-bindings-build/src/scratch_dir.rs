use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

static NEXT_SCRATCH_ID: AtomicU32 = AtomicU32::new(0);

/// A real directory under the system temp dir, removed when the guard drops.
///
/// Build-script logic is filesystem logic, so its tests operate on real
/// directories rather than stand-ins. Names are kept short: cmake and MSBuild
/// nest deeply below them, and Windows still rejects paths over `MAX_PATH`.
pub struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    pub fn new(label: &str) -> Self {
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!("{label}-{}-{unique}", std::process::id()));

        std::fs::create_dir_all(&path).expect("scratch directory must be creatable");

        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::ScratchDir;

    const MAX_WINDOWS_PATH: usize = 260;

    /// Longest suffix observed below a scratch dir, from an MSBuild try-compile:
    /// `cmake-out\build\CMakeFiles\CMakeScratch\TryCompile-xxxxxx\cmTC_NNNNN.dir\Debug\cmTC_NNNNN.tlog\CL.command.1.tlog`
    const NESTED_BUILD_TREE_BUDGET: usize = 120;

    #[test]
    fn scratch_paths_leave_room_for_a_nested_build_tree() {
        let scratch = ScratchDir::new("cmake-build");
        let length = scratch.path().as_os_str().len();

        assert!(
            length + NESTED_BUILD_TREE_BUDGET < MAX_WINDOWS_PATH,
            "scratch path is {length} chars: {}",
            scratch.path().display()
        );
    }

    #[test]
    fn each_scratch_dir_is_distinct_and_removed_on_drop() {
        let first = ScratchDir::new("scratch");
        let path = first.path().to_path_buf();

        assert!(path.is_dir());

        let second = ScratchDir::new("scratch");

        assert_ne!(second.path(), path);

        drop(first);

        assert!(!path.exists());
    }
}
