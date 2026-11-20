use crate::gguf_source::GgufSource;
use crate::model_file::ModelFile;
use crate::model_load_params::ModelLoadParams;
use crate::phase_setup_error::PhaseSetupError;

/// What identifies a loaded model: every trial that shares it runs against one load of the model.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelKey {
    pub model_source: GgufSource,
    pub model_load_params: ModelLoadParams,
}

impl ModelKey {
    /// # Errors
    ///
    /// Returns [`PhaseSetupError::ModelDownload`] when the model file cannot be downloaded.
    pub fn download_model_file(&self) -> Result<ModelFile, PhaseSetupError> {
        let Self { model_source, .. } = *self;

        Ok(ModelFile {
            model_path: model_source.resolve_path().map_err(|cause| {
                PhaseSetupError::ModelDownload {
                    model_source,
                    cause,
                }
            })?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use llama_cpp_params::llama_load_mode::LlamaLoadMode;

    use super::ModelKey;
    use crate::gguf_source::GgufSource;
    use crate::model_load_params::ModelLoadParams;
    use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

    const MODEL_LOAD_PARAMS: ModelLoadParams = ModelLoadParams {
        n_gpu_layers: 0,
        load_mode: LlamaLoadMode::Mmap,
    };

    #[test]
    fn download_model_file_resolves_a_local_source_to_its_path() {
        let model_file = ModelKey {
            model_source: GgufSource::LocalPath("/abs/model.gguf"),
            model_load_params: MODEL_LOAD_PARAMS,
        }
        .download_model_file()
        .expect("a local source resolves without downloading anything");

        assert_eq!(model_file.model_path, PathBuf::from("/abs/model.gguf"));
    }

    #[test]
    fn a_model_that_cannot_be_downloaded_fails_as_a_model_download() {
        assert_eq!(
            ModelKey {
                model_source: GgufSource::HuggingFace {
                    repo: "intentee-test-harness/does-not-exist",
                    file: "no-such-file.gguf",
                },
                model_load_params: MODEL_LOAD_PARAMS,
            }
            .download_model_file()
            .err()
            .map(|setup_failure| setup_failure.kind()),
            Some(PhaseSetupFailureKind::ModelDownload)
        );
    }
}
