use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

static NEXT_SCRATCH_ID: AtomicU32 = AtomicU32::new(0);

/// A real directory under the system temp dir, removed when the guard drops.
///
/// Build-script logic is filesystem logic, so its tests operate on real
/// directories rather than stand-ins.
pub struct ScratchDir {
    path: PathBuf,
}

impl ScratchDir {
    pub fn new(label: &str) -> Self {
        let unique = NEXT_SCRATCH_ID.fetch_add(1, Ordering::Relaxed);
        let path = env::temp_dir().join(format!(
            "llama-cpp-bindings-build-{label}-{}-{unique}",
            std::process::id()
        ));

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
