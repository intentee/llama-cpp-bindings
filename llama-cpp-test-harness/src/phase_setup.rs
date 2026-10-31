use std::sync::Arc;
use std::sync::OnceLock;

use anyhow::Error;
use anyhow::Result;

use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::load_key::LoadKey;
use crate::phase_model_files::PhaseModelFiles;
use crate::phase_state::PhaseState;

pub struct PhaseSetup {
    key: LoadKey,
    backend: Arc<LlamaBackend>,
    model_files: OnceLock<Result<PhaseModelFiles>>,
    phase_state: OnceLock<Result<PhaseState>>,
}

impl PhaseSetup {
    #[must_use]
    pub const fn new(key: LoadKey, backend: Arc<LlamaBackend>) -> Self {
        Self {
            key,
            backend,
            model_files: OnceLock::new(),
            phase_state: OnceLock::new(),
        }
    }

    /// # Errors
    ///
    /// Returns the failure of downloading the model or the mmproj file. The outcome is cached, so
    /// the files are downloaded at most once per phase.
    pub fn download_model_files(&self) -> Result<&PhaseModelFiles, &Error> {
        self.model_files
            .get_or_init(|| self.key.download_model_files())
            .as_ref()
    }

    /// # Errors
    ///
    /// Returns the cached download failure, or the failure of loading the downloaded files into
    /// llama.cpp. The outcome is cached, so the model is loaded at most once per phase.
    pub fn load_phase_state(&self) -> Result<&PhaseState, &Error> {
        let model_files = self.download_model_files()?;

        self.phase_state
            .get_or_init(|| model_files.load_phase_state(&self.backend, self.key.model_load_params))
            .as_ref()
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;
    use std::sync::Arc;
    use std::sync::PoisonError;

    use llama_cpp_bindings::llama_backend::LlamaBackend;
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;

    use super::PhaseSetup;
    use crate::gguf_source::GgufSource;
    use crate::load_key::LoadKey;
    use crate::model_load_params::ModelLoadParams;
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    const NON_GGUF_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    fn key_for_model(model_source: GgufSource) -> LoadKey {
        LoadKey {
            model_source,
            mmproj_source: None,
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
            .download_model_files()
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
