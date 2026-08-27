use std::path::PathBuf;

#[must_use]
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::fixtures_dir;

    #[test]
    fn resolves_to_the_fixtures_directory_inside_the_manifest() {
        let expected = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");

        assert_eq!(fixtures_dir(), expected);
        assert!(
            fixtures_dir().is_dir(),
            "the fixtures directory the multimodal tests read from must exist"
        );
    }
}
