use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::mtmd_context::MtmdContext;
use llama_cpp_bindings::mtmd::mtmd_context_params::MtmdContextParams;

use crate::model_load_params::ModelLoadParams;
use crate::phase_state::PhaseState;

#[derive(Debug)]
pub struct PhaseModelFiles {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
}

impl PhaseModelFiles {
    /// # Errors
    ///
    /// Returns an error if loading the model into llama.cpp fails, or initializing the
    /// `MtmdContext` from the mmproj file fails.
    pub fn load_phase_state(
        &self,
        backend: &Arc<LlamaBackend>,
        model_load_params: ModelLoadParams,
    ) -> Result<PhaseState> {
        let Self {
            model_path,
            mmproj_path,
        } = self;
        let model_params = model_load_params.into_llama_model_params();
        let model = LlamaModel::load_from_file(backend, model_path, &model_params)?;

        let mtmd_context = match mmproj_path {
            Some(mmproj_path) => {
                let mmproj_path_str = mmproj_path.to_string_lossy();
                let params = MtmdContextParams::default();
                Some(MtmdContext::init_from_file(
                    mmproj_path_str.as_ref(),
                    &model,
                    &params,
                )?)
            }
            None => None,
        };

        Ok(PhaseState {
            mtmd_context,
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

    use llama_cpp_bindings::error::llama_model_load_error::LlamaModelLoadError;
    use llama_cpp_bindings::llama_backend::LlamaBackend;
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use super::PhaseModelFiles;
    use crate::model_load_params::ModelLoadParams;
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    #[test]
    fn load_phase_state_propagates_model_load_failure() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let model_files = PhaseModelFiles {
            model_path: PathBuf::from(NON_GGUF_PATH),
            mmproj_path: None,
        };

        let load_failure = model_files
            .load_phase_state(
                &backend,
                ModelLoadParams {
                    n_gpu_layers: 0,
                    load_mode: LlamaLoadMode::Mmap,
                },
            )
            .expect_err("a model file that is not GGUF must fail to load");

        assert_eq!(
            load_failure.downcast_ref::<LlamaModelLoadError>(),
            Some(&LlamaModelLoadError::Unloadable)
        );
    }
}
