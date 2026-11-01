use std::path::PathBuf;

#[must_use]
pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

#[cfg(test)]
mod tests {
    use super::fixtures_dir;

    #[test]
    fn resolves_to_the_directory_holding_every_media_fixture() {
        assert_eq!(
            ["llamas.jpg", "orange_cat.wav", "quick_brown_fox.wav"]
                .map(|file_name| fixtures_dir().join(file_name).is_file()),
            [true, true, true]
        );
    }
}
