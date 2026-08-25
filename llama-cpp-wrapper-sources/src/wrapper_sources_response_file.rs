use std::fs;
use std::path::PathBuf;

use crate::wrapper_source_paths::wrapper_source_paths;
use crate::wrapper_sources_error::WrapperSourcesError;

pub struct WrapperSourcesResponseFile {
    pub output_path: PathBuf,
    pub sys_dir: PathBuf,
}

impl WrapperSourcesResponseFile {
    /// # Errors
    ///
    /// Returns [`WrapperSourcesError`] when the response file cannot be written.
    pub fn write(&self) -> Result<(), WrapperSourcesError> {
        let mut contents = wrapper_source_paths(&self.sys_dir)
            .into_iter()
            .map(|source_path| format!("\"{}\"", source_path.display()))
            .collect::<Vec<String>>()
            .join("\n");

        contents.push('\n');

        fs::write(&self.output_path, contents).map_err(|source| WrapperSourcesError::Write {
            path: self.output_path.clone(),
            source,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use super::WrapperSourcesResponseFile;
    use crate::wrapper_sources_error::WrapperSourcesError;

    #[test]
    fn writing_under_a_missing_directory_reports_the_path() {
        let output_path = "/nonexistent-directory/wrapper_sources.rsp";
        let response_file = WrapperSourcesResponseFile {
            output_path: PathBuf::from(output_path),
            sys_dir: PathBuf::from("/repo/llama-cpp-bindings-sys"),
        };

        assert!(matches!(
            response_file.write(),
            Err(WrapperSourcesError::Write { ref path, .. }) if path == Path::new(output_path)
        ));
    }
}
