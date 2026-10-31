use anyhow::Result;

use crate::gguf_source::GgufSource;
use crate::model_load_params::ModelLoadParams;
use crate::phase_model_files::PhaseModelFiles;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoadKey {
    pub model_source: GgufSource,
    pub mmproj_source: Option<GgufSource>,
    pub model_load_params: ModelLoadParams,
}

impl LoadKey {
    /// # Errors
    ///
    /// Returns an error if downloading the model file or the mmproj file fails.
    pub fn download_model_files(&self) -> Result<PhaseModelFiles> {
        Ok(PhaseModelFiles {
            model_path: self.model_source.resolve_path()?,
            mmproj_path: self
                .mmproj_source
                .map(GgufSource::resolve_path)
                .transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use crate::gguf_source::GgufSource;
    use crate::model_load_params::ModelLoadParams;

    use super::LoadKey;

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
        other.model_load_params.n_gpu_layers = 999;

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
    fn download_model_files_propagates_mmproj_download_failure() {
        let missing_mmproj_source = GgufSource::HuggingFace {
            repo: "intentee-test-harness/does-not-exist",
            file: "no-such-mmproj.gguf",
        };
        let mut key = baseline();
        key.model_source = GgufSource::LocalPath("/abs/model.gguf");
        key.mmproj_source = Some(missing_mmproj_source);

        let download_failure = key
            .download_model_files()
            .expect_err("an mmproj from a repository that does not exist must fail to download");

        assert_eq!(
            download_failure.downcast_ref::<GgufSource>(),
            Some(&missing_mmproj_source)
        );
    }
}
