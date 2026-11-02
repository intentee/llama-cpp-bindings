use std::path::PathBuf;

use hf_hub::api::sync::ApiBuilder;
use hf_hub::api::sync::ApiError;

/// # Errors
///
/// Returns the [`ApiError`] of building the HF client or of downloading the file, e.g. when the
/// repository or file does not exist, or network access fails.
pub fn download_model(repo: &str, file: &str) -> Result<PathBuf, ApiError> {
    ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .and_then(|api| api.model(repo.to_owned()).get(file))
}
