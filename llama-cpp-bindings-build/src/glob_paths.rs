use std::path::Path;
use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum GlobPathsError {
    #[error("invalid glob pattern {pattern:?}: {source}")]
    InvalidPattern {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },
    #[error("glob entry failed for pattern {pattern:?}: {source}")]
    EntryError {
        pattern: String,
        #[source]
        source: glob::GlobError,
    },
    #[error("no files matched glob pattern {pattern:?}")]
    NoMatches { pattern: String },
}

pub fn collect_paths(directory: &Path, file_pattern: &str) -> Result<Vec<PathBuf>, GlobPathsError> {
    let escaped = glob::Pattern::escape(&directory.to_string_lossy());
    let pattern = Path::new(&escaped).join(file_pattern);
    let pattern = pattern.to_string_lossy();
    let pattern = pattern.as_ref();

    let entries = glob::glob(pattern).map_err(|source| GlobPathsError::InvalidPattern {
        pattern: pattern.to_string(),
        source,
    })?;

    let mut paths = Vec::new();

    for entry in entries {
        let path = entry.map_err(|source| GlobPathsError::EntryError {
            pattern: pattern.to_string(),
            source,
        })?;

        paths.push(path);
    }

    if paths.is_empty() {
        return Err(GlobPathsError::NoMatches {
            pattern: pattern.to_string(),
        });
    }

    paths.sort();

    Ok(paths)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::scratch_dir::ScratchDir;

    use super::GlobPathsError;
    use super::collect_paths;

    #[test]
    fn matching_pattern_returns_sorted_paths() {
        let paths = collect_paths(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
            "*.rs",
        )
        .expect("sources must match");

        assert!(paths.len() > 1, "the crate has several source files");

        let mut sorted = paths.clone();
        sorted.sort();

        assert_eq!(paths, sorted, "collect_paths must return sorted paths");
        assert!(
            paths.iter().any(|path| path.ends_with("glob_paths.rs")),
            "this very file must be among the matches"
        );
    }

    #[test]
    fn an_invalid_file_pattern_reports_the_pattern() {
        let scratch = ScratchDir::new("glob-badpattern");

        let error =
            collect_paths(scratch.path(), "**b/c").expect_err("malformed recursive glob must fail");

        assert!(matches!(error, GlobPathsError::InvalidPattern { .. }));
        assert!(error.to_string().contains("invalid glob pattern"));
    }

    #[test]
    fn glob_metacharacters_in_the_directory_are_matched_literally() {
        let scratch = ScratchDir::new("glob-metachars");
        let awkward = scratch.path().join("a[b]c");
        std::fs::create_dir_all(&awkward).expect("directory must be creatable");
        std::fs::write(awkward.join("found.txt"), b"x").expect("file must be writable");

        let paths = collect_paths(&awkward, "*.txt").expect("the directory must match literally");

        assert_eq!(paths.len(), 1, "got: {paths:?}");
        assert!(paths[0].ends_with("found.txt"));
    }

    #[test]
    fn a_pattern_matching_nothing_reports_no_matches() {
        let scratch = ScratchDir::new("glob-empty");

        let error =
            collect_paths(scratch.path(), "*.nothing-here").expect_err("empty directory must fail");

        assert!(matches!(error, GlobPathsError::NoMatches { .. }));
        assert!(error.to_string().contains("no files matched"));
    }

    #[cfg(unix)]
    #[test]
    fn an_unreadable_directory_reports_an_entry_error() {
        use std::os::unix::fs::PermissionsExt;

        let scratch = ScratchDir::new("glob-unreadable");
        let locked = scratch.path().join("locked");
        std::fs::create_dir_all(&locked).expect("nested directory must be creatable");
        std::fs::write(locked.join("inner.txt"), b"x").expect("file must be writable");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("permissions must be settable");

        let outcome = collect_paths(scratch.path(), "*/*");

        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))
            .expect("permissions must be restorable for cleanup");

        let error = outcome.expect_err("an unreadable directory must surface an entry error");

        assert!(matches!(error, GlobPathsError::EntryError { .. }));
        assert!(error.to_string().contains("glob entry failed"));
    }
}
