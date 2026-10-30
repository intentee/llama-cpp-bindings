use std::path::PathBuf;

use anyhow::Result;

/// # Errors
///
/// Returns an error if the HF client cannot be built or the file cannot be downloaded
/// (e.g., the repo or file does not exist, or network access fails).
pub fn download_model(repo: &str, file: &str) -> Result<PathBuf> {
    Ok(hf_hub::api::sync::ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .and_then(|api| api.model(repo.to_owned()).get(file))?)
}

#[cfg(test)]
mod tests {
    use hf_hub::api::sync::ApiError;

    use super::download_model;

    #[test]
    fn missing_file_in_real_repo_returns_error() {
        let download_failure =
            download_model("unsloth/Qwen3.5-0.8B-GGUF", "this-file-does-not-exist.gguf")
                .expect_err("a file missing from an existing repository must fail to download");

        assert!(download_failure.is::<ApiError>());
    }
}
