use hf_hub::api::sync::ApiError;
use std::path::PathBuf;

use llama_cpp_model::model_error::ModelError;
use llama_cpp_mtmd::mtmd_error::MtmdError;

use crate::gguf_source::GgufSource;
use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

#[derive(Debug, thiserror::Error)]
pub enum PhaseSetupError {
    #[error("the model {model_source} could not be downloaded: {cause}")]
    ModelDownload {
        model_source: GgufSource,
        #[source]
        cause: ApiError,
    },
    #[error("the mmproj {mmproj_source} could not be downloaded: {cause}")]
    MmprojDownload {
        mmproj_source: GgufSource,
        #[source]
        cause: ApiError,
    },
    #[error("the mmproj path {} is not Unicode", mmproj_path.display())]
    MmprojPathNotUnicode { mmproj_path: PathBuf },
    #[error("the model {} could not be loaded: {cause}", model_path.display())]
    ModelLoad {
        model_path: PathBuf,
        #[source]
        cause: ModelError,
    },
    #[error("the mmproj {} could not be loaded: {cause}", mmproj_path.display())]
    MmprojLoad {
        mmproj_path: PathBuf,
        #[source]
        cause: MtmdError,
    },
}

impl PhaseSetupError {
    #[must_use]
    pub const fn kind(&self) -> PhaseSetupFailureKind {
        match self {
            Self::ModelDownload { .. } => PhaseSetupFailureKind::ModelDownload,
            Self::MmprojDownload { .. } => PhaseSetupFailureKind::MmprojDownload,
            Self::MmprojPathNotUnicode { .. } => PhaseSetupFailureKind::MmprojPathNotUnicode,
            Self::ModelLoad { .. } => PhaseSetupFailureKind::ModelLoad,
            Self::MmprojLoad { .. } => PhaseSetupFailureKind::MmprojLoad,
        }
    }
}
