use crate::gguf_source::GgufSource;
use crate::model_key::ModelKey;
use crate::model_load_params::ModelLoadParams;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LoadKey {
    pub model_source: GgufSource,
    pub mmproj_source: Option<GgufSource>,
    pub model_load_params: ModelLoadParams,
}

impl LoadKey {
    #[must_use]
    pub const fn model_key(&self) -> ModelKey {
        let Self {
            model_source,
            model_load_params,
            ..
        } = *self;

        ModelKey {
            model_source,
            model_load_params,
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_params::llama_load_mode::LlamaLoadMode;

    use super::LoadKey;
    use crate::gguf_source::GgufSource;
    use crate::model_load_params::ModelLoadParams;

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
    fn keys_differing_only_in_their_mmproj_source_share_a_model_key() {
        let mut other = baseline();
        other.mmproj_source = Some(GgufSource::HuggingFace {
            repo: "repo",
            file: "mmproj-F16.gguf",
        });

        assert_eq!(baseline().model_key(), other.model_key());
    }
}
