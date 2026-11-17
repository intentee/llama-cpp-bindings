use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_model::llama_model::LlamaModel;

use crate::model_load_params::ModelLoadParams;
use crate::phase_setup_error::PhaseSetupError;
use crate::phase_state::PhaseState;

#[derive(Debug)]
pub struct ModelFile {
    pub model_path: PathBuf,
}

impl ModelFile {
    /// # Errors
    ///
    /// Returns [`PhaseSetupError::ModelLoad`] when llama.cpp cannot load the model.
    pub fn load_phase_state(
        &self,
        backend: &Arc<LlamaBackend>,
        model_load_params: ModelLoadParams,
    ) -> Result<PhaseState, PhaseSetupError> {
        let Self { model_path } = self;
        let model = LlamaModel::load_from_file(
            backend,
            model_path,
            &model_load_params.into_llama_model_params(),
        )
        .map_err(|cause| PhaseSetupError::ModelLoad {
            model_path: model_path.clone(),
            cause,
        })?;

        Ok(PhaseState {
            model,
            backend: Arc::clone(backend),
            model_path: model_path.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::PoisonError;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_params::llama_load_mode::LlamaLoadMode;

    use super::ModelFile;
    use crate::backend_init_gate::BACKEND_INIT_GATE;
    use crate::model_load_params::ModelLoadParams;
    use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    #[test]
    fn a_model_file_that_is_not_gguf_fails_as_a_model_load() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));

        assert_eq!(
            ModelFile {
                model_path: PathBuf::from(NON_GGUF_PATH),
            }
            .load_phase_state(
                &backend,
                ModelLoadParams {
                    n_gpu_layers: 0,
                    load_mode: LlamaLoadMode::Mmap,
                },
            )
            .err()
            .map(|setup_failure| setup_failure.kind()),
            Some(PhaseSetupFailureKind::ModelLoad)
        );
    }
}
