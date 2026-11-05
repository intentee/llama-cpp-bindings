use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::mtmd::mtmd_context::MtmdContext;
use llama_cpp_bindings::mtmd::mtmd_context_params::MtmdContextParams;

use crate::model_load_params::ModelLoadParams;
use crate::phase_setup_error::PhaseSetupError;
use crate::phase_state::PhaseState;

fn unicode_mmproj_path(mmproj_path: &Path) -> Result<&str, PhaseSetupError> {
    mmproj_path
        .to_str()
        .ok_or_else(|| PhaseSetupError::MmprojPathNotUnicode {
            mmproj_path: mmproj_path.to_path_buf(),
        })
}

#[derive(Debug)]
pub struct PhaseModelFiles {
    pub model_path: PathBuf,
    pub mmproj_path: Option<PathBuf>,
}

impl PhaseModelFiles {
    /// # Errors
    ///
    /// Returns [`PhaseSetupError::MmprojPathNotUnicode`] before loading anything when the mmproj
    /// path cannot be handed to llama.cpp, [`PhaseSetupError::ModelLoad`] when llama.cpp cannot
    /// load the model, or [`PhaseSetupError::MmprojLoad`] when it cannot load the mmproj.
    pub fn load_phase_state(
        &self,
        backend: &Arc<LlamaBackend>,
        model_load_params: ModelLoadParams,
    ) -> Result<PhaseState, PhaseSetupError> {
        let Self {
            model_path,
            mmproj_path,
        } = self;
        let unicode_mmproj_path = mmproj_path
            .as_deref()
            .map(unicode_mmproj_path)
            .transpose()?;
        let model = LlamaModel::load_from_file(
            backend,
            model_path,
            &model_load_params.into_llama_model_params(),
        )
        .map_err(|cause| PhaseSetupError::ModelLoad {
            model_path: model_path.clone(),
            cause,
        })?;
        let mtmd_context = unicode_mmproj_path
            .map(|unicode_mmproj_path| {
                MtmdContext::init_from_file(
                    unicode_mmproj_path,
                    &model,
                    &MtmdContextParams::default(),
                )
                .map_err(|cause| PhaseSetupError::MmprojLoad {
                    mmproj_path: PathBuf::from(unicode_mmproj_path),
                    cause,
                })
            })
            .transpose()?;

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
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::PoisonError;

    use llama_cpp_bindings::llama_backend::LlamaBackend;
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use super::PhaseModelFiles;
    use crate::backend_init_gate::BACKEND_INIT_GATE;
    use crate::model_load_params::ModelLoadParams;
    use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    fn load_failure_kind(model_files: &PhaseModelFiles) -> Option<PhaseSetupFailureKind> {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));

        model_files
            .load_phase_state(
                &backend,
                ModelLoadParams {
                    n_gpu_layers: 0,
                    load_mode: LlamaLoadMode::Mmap,
                },
            )
            .err()
            .map(|setup_failure| setup_failure.kind())
    }

    #[test]
    fn a_model_file_that_is_not_gguf_fails_as_a_model_load() {
        assert_eq!(
            load_failure_kind(&PhaseModelFiles {
                model_path: PathBuf::from(NON_GGUF_PATH),
                mmproj_path: None,
            }),
            Some(PhaseSetupFailureKind::ModelLoad)
        );
    }

    #[test]
    fn a_non_unicode_mmproj_path_is_refused_before_the_model_loads() {
        assert_eq!(
            load_failure_kind(&PhaseModelFiles {
                model_path: PathBuf::from(NON_GGUF_PATH),
                mmproj_path: Some(PathBuf::from(OsString::from_vec(vec![0xff]))),
            }),
            Some(PhaseSetupFailureKind::MmprojPathNotUnicode)
        );
    }
}
