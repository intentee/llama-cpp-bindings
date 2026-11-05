use crate::gguf_source::GgufSource;
use crate::model_load_params::ModelLoadParams;
use crate::phase_model_files::PhaseModelFiles;
use crate::phase_setup_error::PhaseSetupError;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoadKey {
    pub model_source: GgufSource,
    pub mmproj_source: Option<GgufSource>,
    pub model_load_params: ModelLoadParams,
}

impl LoadKey {
    /// # Errors
    ///
    /// Returns [`PhaseSetupError::ModelDownload`] or [`PhaseSetupError::MmprojDownload`] when the
    /// model or the mmproj file cannot be downloaded.
    pub fn download_model_files(&self) -> Result<PhaseModelFiles, PhaseSetupError> {
        let Self {
            model_source,
            mmproj_source,
            ..
        } = *self;

        Ok(PhaseModelFiles {
            model_path: model_source.resolve_path().map_err(|cause| {
                PhaseSetupError::ModelDownload {
                    model_source,
                    cause,
                }
            })?,
            mmproj_path: mmproj_source
                .map(|mmproj_source| {
                    mmproj_source
                        .resolve_path()
                        .map_err(|cause| PhaseSetupError::MmprojDownload {
                            mmproj_source,
                            cause,
                        })
                })
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use super::LoadKey;
    use crate::gguf_source::GgufSource;
    use crate::model_load_params::ModelLoadParams;
    use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

    fn baseline() -> LoadKey {
        LoadKey {
            model_source: GgufSource::HuggingFace {
                repo: "repo",
                file: "file",
            },
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        }
    }

    #[test]
    fn identical_keys_compare_equal() {
        assert_eq!(baseline(), baseline());
    }

    #[test]
    fn different_model_sources_compare_unequal() {
        let mut other = baseline();
        other.model_source = GgufSource::HuggingFace {
            repo: "other",
            file: "file",
        };

        assert_ne!(baseline(), other);
    }

    #[test]
    fn huggingface_and_local_path_compare_unequal() {
        let mut other = baseline();
        other.model_source = GgufSource::LocalPath("/some/local.gguf");

        assert_ne!(baseline(), other);
    }

    #[test]
    fn different_mmproj_sources_compare_unequal() {
        let mut other = baseline();
        other.mmproj_source = Some(GgufSource::HuggingFace {
            repo: "repo",
            file: "mmproj-F16.gguf",
        });

        assert_ne!(baseline(), other);
    }

    #[test]
    fn different_model_load_params_compare_unequal() {
        let mut other = baseline();
        other.model_load_params.n_gpu_layers = -1;

        assert_ne!(baseline(), other);
    }

    #[test]
    fn download_model_files_resolves_local_sources_to_their_paths() {
        let mut key = baseline();
        key.model_source = GgufSource::LocalPath("/abs/model.gguf");
        key.mmproj_source = Some(GgufSource::LocalPath("/abs/mmproj.gguf"));

        let model_files = key
            .download_model_files()
            .expect("local sources resolve without downloading anything");

        assert_eq!(model_files.model_path, PathBuf::from("/abs/model.gguf"));
        assert_eq!(
            model_files.mmproj_path,
            Some(PathBuf::from("/abs/mmproj.gguf"))
        );
    }

    #[test]
    fn a_model_that_cannot_be_downloaded_fails_as_a_model_download() {
        let mut key = baseline();
        key.model_source = GgufSource::HuggingFace {
            repo: "intentee-test-harness/does-not-exist",
            file: "no-such-file.gguf",
        };

        assert_eq!(
            key.download_model_files()
                .err()
                .map(|setup_failure| setup_failure.kind()),
            Some(PhaseSetupFailureKind::ModelDownload)
        );
    }

    #[test]
    fn an_mmproj_that_cannot_be_downloaded_fails_as_an_mmproj_download() {
        let mut key = baseline();
        key.model_source = GgufSource::LocalPath("/abs/model.gguf");
        key.mmproj_source = Some(GgufSource::HuggingFace {
            repo: "intentee-test-harness/does-not-exist",
            file: "no-such-mmproj.gguf",
        });

        assert_eq!(
            key.download_model_files()
                .err()
                .map(|setup_failure| setup_failure.kind()),
            Some(PhaseSetupFailureKind::MmprojDownload)
        );
    }
}
