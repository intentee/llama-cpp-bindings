use std::sync::Arc;
use std::sync::OnceLock;

use llama_cpp_backend::llama_backend::LlamaBackend;

use crate::model_file::ModelFile;
use crate::model_key::ModelKey;
use crate::phase_setup_error::PhaseSetupError;
use crate::phase_state::PhaseState;

pub struct PhaseSetup {
    key: ModelKey,
    backend: Arc<LlamaBackend>,
    model_file: OnceLock<Result<ModelFile, PhaseSetupError>>,
    phase_state: OnceLock<Result<PhaseState, PhaseSetupError>>,
}

impl PhaseSetup {
    #[must_use]
    pub const fn new(key: ModelKey, backend: Arc<LlamaBackend>) -> Self {
        Self {
            key,
            backend,
            model_file: OnceLock::new(),
            phase_state: OnceLock::new(),
        }
    }

    /// # Errors
    ///
    /// Returns the failure of downloading the model file. The outcome is cached, so the file is
    /// downloaded at most once per phase.
    pub fn download_model_file(&self) -> Result<&ModelFile, &PhaseSetupError> {
        self.model_file
            .get_or_init(|| self.key.download_model_file())
            .as_ref()
    }

    /// # Errors
    ///
    /// Returns the cached download failure, or the failure of loading the downloaded model into
    /// llama.cpp. The outcome is cached, so the model is loaded at most once per phase.
    pub fn load_phase_state(&self) -> Result<&PhaseState, &PhaseSetupError> {
        let model_file = self.download_model_file()?;

        self.phase_state
            .get_or_init(|| model_file.load_phase_state(&self.backend, self.key.model_load_params))
            .as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::Arc;
    use std::sync::PoisonError;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_params::llama_load_mode::LlamaLoadMode;

    use super::PhaseSetup;
    use crate::backend_init_gate::BACKEND_INIT_GATE;
    use crate::gguf_source::GgufSource;
    use crate::model_key::ModelKey;
    use crate::model_load_params::ModelLoadParams;

    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    fn key_for_model(model_source: GgufSource) -> ModelKey {
        ModelKey {
            model_source,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        }
    }

    #[test]
    fn load_phase_state_returns_the_cached_download_failure() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let setup = PhaseSetup::new(
            key_for_model(GgufSource::HuggingFace {
                repo: "intentee-test-harness/does-not-exist",
                file: "no-such-file.gguf",
            }),
            backend,
        );

        let download_failure = setup
            .download_model_file()
            .expect_err("a model from a repository that does not exist must fail to download");
        let setup_failure = setup
            .load_phase_state()
            .expect_err("a phase whose model failed to download must fail to set up");

        assert!(ptr::eq(download_failure, setup_failure));
    }

    #[test]
    fn load_phase_state_loads_once_and_caches_the_outcome() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let setup = PhaseSetup::new(key_for_model(GgufSource::LocalPath(NON_GGUF_PATH)), backend);

        let first_failure = setup
            .load_phase_state()
            .expect_err("a model file that is not GGUF must fail to load");
        let second_failure = setup
            .load_phase_state()
            .expect_err("a cached load failure must be reported again");

        assert!(ptr::eq(first_failure, second_failure));
    }
}
