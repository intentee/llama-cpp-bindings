use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;
use llama_cpp_bindings::model::params::LlamaModelParams;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ModelLoadParams {
    pub n_gpu_layers: i32,
    pub load_mode: LlamaLoadMode,
}

impl ModelLoadParams {
    #[must_use]
    pub fn into_llama_model_params(self) -> LlamaModelParams {
        let Self {
            n_gpu_layers,
            load_mode,
        } = self;
        LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_load_mode(load_mode)
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use super::ModelLoadParams;

    #[test]
    fn into_llama_model_params_carries_all_three_fields() {
        let params = ModelLoadParams {
            n_gpu_layers: 7,
            load_mode: LlamaLoadMode::Mlock,
        }
        .into_llama_model_params();

        assert_eq!(params.n_gpu_layers(), 7);
        assert_eq!(params.load_mode(), Ok(LlamaLoadMode::Mlock));
    }

    #[test]
    fn identical_values_compare_equal() {
        let one = ModelLoadParams {
            n_gpu_layers: 1,
            load_mode: LlamaLoadMode::Mmap,
        };
        let two = ModelLoadParams {
            n_gpu_layers: 1,
            load_mode: LlamaLoadMode::Mmap,
        };

        assert_eq!(one, two);
    }

    #[test]
    fn differing_n_gpu_layers_compare_unequal() {
        let one = ModelLoadParams {
            n_gpu_layers: 1,
            load_mode: LlamaLoadMode::Mmap,
        };
        let two = ModelLoadParams {
            n_gpu_layers: 2,
            load_mode: LlamaLoadMode::Mmap,
        };

        assert_ne!(one, two);
    }
}
