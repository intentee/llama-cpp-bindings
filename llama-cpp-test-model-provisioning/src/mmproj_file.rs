use std::path::PathBuf;

use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_context_params::MtmdContextParams;

use crate::phase_setup_error::PhaseSetupError;

#[derive(Debug)]
pub struct MmprojFile {
    pub unicode_mmproj_path: String,
}

impl MmprojFile {
    /// # Errors
    ///
    /// Returns [`PhaseSetupError::MmprojPathNotUnicode`] when the mmproj path cannot be handed to
    /// llama.cpp.
    pub fn from_path(mmproj_path: PathBuf) -> Result<Self, PhaseSetupError> {
        mmproj_path
            .into_os_string()
            .into_string()
            .map(|unicode_mmproj_path| Self {
                unicode_mmproj_path,
            })
            .map_err(|mmproj_path| PhaseSetupError::MmprojPathNotUnicode {
                mmproj_path: PathBuf::from(mmproj_path),
            })
    }

    /// # Errors
    ///
    /// Returns [`PhaseSetupError::MmprojLoad`] when llama.cpp cannot load the mmproj for `model`.
    pub fn load_mtmd_context(&self, model: &LlamaModel) -> Result<MtmdContext, PhaseSetupError> {
        let Self {
            unicode_mmproj_path,
        } = self;

        MtmdContext::init_from_file(unicode_mmproj_path, model, &MtmdContextParams::default())
            .map_err(|cause| PhaseSetupError::MmprojLoad {
                mmproj_path: PathBuf::from(unicode_mmproj_path),
                cause,
            })
    }
}
