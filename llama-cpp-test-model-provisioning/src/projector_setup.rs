use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;

use llama_cpp_mtmd::mtmd_context::MtmdContext;

use crate::gguf_source::GgufSource;
use crate::mmproj_file::MmprojFile;
use crate::phase_setup::PhaseSetup;
use crate::phase_setup_error::PhaseSetupError;
use crate::projector_state::ProjectorState;

fn lock_projector<'setup, TProjector>(
    projector: &'setup Mutex<TProjector>,
    poisoning: &'setup OnceLock<PhaseSetupError>,
    mmproj_source: GgufSource,
) -> Result<MutexGuard<'setup, TProjector>, &'setup PhaseSetupError> {
    projector.lock().map_err(|_poisoned_projector| {
        poisoning.get_or_init(|| PhaseSetupError::MmprojPoisoned { mmproj_source })
    })
}

pub struct ProjectorSetup {
    mmproj_source: GgufSource,
    mmproj_file: OnceLock<Result<MmprojFile, PhaseSetupError>>,
    mtmd_context: OnceLock<Result<Mutex<MtmdContext>, PhaseSetupError>>,
    mtmd_context_poisoning: OnceLock<PhaseSetupError>,
    phase_setup: Arc<PhaseSetup>,
}

impl ProjectorSetup {
    #[must_use]
    pub const fn new(mmproj_source: GgufSource, phase_setup: Arc<PhaseSetup>) -> Self {
        Self {
            mmproj_source,
            mmproj_file: OnceLock::new(),
            mtmd_context: OnceLock::new(),
            mtmd_context_poisoning: OnceLock::new(),
            phase_setup,
        }
    }

    /// # Errors
    ///
    /// Returns [`PhaseSetupError::MmprojDownload`] when the mmproj file cannot be downloaded, or
    /// [`PhaseSetupError::MmprojPathNotUnicode`] when its path cannot be handed to llama.cpp. The
    /// outcome is cached, so the file is downloaded at most once per phase.
    pub fn download_mmproj_file(&self) -> Result<&MmprojFile, &PhaseSetupError> {
        self.mmproj_file
            .get_or_init(|| {
                self.mmproj_source
                    .resolve_path()
                    .map_err(|cause| PhaseSetupError::MmprojDownload {
                        mmproj_source: self.mmproj_source,
                        cause,
                    })
                    .and_then(MmprojFile::from_path)
            })
            .as_ref()
    }

    /// Loads the projector and locks it for as long as the returned state lives, since only one
    /// trial at a time may use it.
    ///
    /// # Errors
    ///
    /// Returns, in this order, the cached failure of downloading the model, of downloading the
    /// mmproj, of loading the model, or of loading the mmproj into llama.cpp, or
    /// [`PhaseSetupError::MmprojPoisoned`] once a trial panicked while it held the projector.
    /// Every outcome is cached, so the mmproj is loaded at most once per phase, into the one model
    /// the phase keeps loaded.
    pub fn load_projector_state(&self) -> Result<ProjectorState<'_>, &PhaseSetupError> {
        self.phase_setup.download_model_file()?;

        let mmproj_file = self.download_mmproj_file()?;
        let phase_state = self.phase_setup.load_phase_state()?;
        let projector = self
            .mtmd_context
            .get_or_init(|| {
                mmproj_file
                    .load_mtmd_context(&phase_state.model)
                    .map(Mutex::new)
            })
            .as_ref()?;

        lock_projector(projector, &self.mtmd_context_poisoning, self.mmproj_source).map(
            |mtmd_context| ProjectorState {
                phase_state,
                mtmd_context,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;
    use std::ptr;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::sync::PoisonError;
    use std::thread;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_params::llama_load_mode::LlamaLoadMode;

    use super::ProjectorSetup;
    use super::lock_projector;
    use crate::backend_init_gate::BACKEND_INIT_GATE;
    use crate::gguf_source::GgufSource;
    use crate::mmproj_file::MmprojFile;
    use crate::model_key::ModelKey;
    use crate::model_load_params::ModelLoadParams;
    use crate::phase_setup::PhaseSetup;
    use crate::phase_setup_error::PhaseSetupError;
    use crate::phase_setup_failure_kind::PhaseSetupFailureKind;

    const NON_GGUF_MODEL: GgufSource =
        GgufSource::LocalPath(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));

    const MISSING_MODEL: GgufSource = GgufSource::HuggingFace {
        repo: "intentee-test-harness/does-not-exist",
        file: "no-such-file.gguf",
    };

    const LOCAL_MMPROJ: GgufSource = GgufSource::LocalPath("/abs/mmproj.gguf");

    const MISSING_MMPROJ: GgufSource = GgufSource::HuggingFace {
        repo: "intentee-test-harness/does-not-exist",
        file: "no-such-mmproj.gguf",
    };

    fn with_phase_setup<TResult, TUse: FnOnce(Arc<PhaseSetup>) -> TResult>(
        model_source: GgufSource,
        use_phase_setup: TUse,
    ) -> TResult {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));

        use_phase_setup(Arc::new(PhaseSetup::new(
            ModelKey {
                model_source,
                model_load_params: ModelLoadParams {
                    n_gpu_layers: 0,
                    load_mode: LlamaLoadMode::Mmap,
                },
            },
            backend,
        )))
    }

    #[test]
    fn download_mmproj_file_resolves_a_local_source_to_its_path() {
        with_phase_setup(NON_GGUF_MODEL, |phase_setup| {
            let projector_setup = ProjectorSetup::new(LOCAL_MMPROJ, phase_setup);

            assert_eq!(
                projector_setup
                    .download_mmproj_file()
                    .expect("a local source resolves without downloading anything")
                    .unicode_mmproj_path,
                "/abs/mmproj.gguf"
            );
        });
    }

    #[test]
    fn a_model_that_cannot_be_downloaded_fails_the_projector_with_the_cached_model_download() {
        with_phase_setup(MISSING_MODEL, |phase_setup| {
            let projector_setup = ProjectorSetup::new(MISSING_MMPROJ, Arc::clone(&phase_setup));

            let model_download_failure = phase_setup
                .download_model_file()
                .expect_err("a model from a repository that does not exist must fail to download");
            let projector_failure = projector_setup
                .load_projector_state()
                .expect_err("a projector whose model failed to download must fail to set up");

            assert!(ptr::eq(model_download_failure, projector_failure));
        });
    }

    #[test]
    fn an_mmproj_that_cannot_be_downloaded_fails_once_as_a_cached_mmproj_download() {
        with_phase_setup(NON_GGUF_MODEL, |phase_setup| {
            let projector_setup = ProjectorSetup::new(MISSING_MMPROJ, phase_setup);

            let download_failure = projector_setup
                .download_mmproj_file()
                .expect_err("an mmproj from a repository that does not exist must fail");
            let projector_failure = projector_setup
                .load_projector_state()
                .expect_err("a projector whose mmproj failed to download must fail to set up");

            assert_eq!(
                download_failure.kind(),
                PhaseSetupFailureKind::MmprojDownload
            );
            assert!(ptr::eq(download_failure, projector_failure));
        });
    }

    #[test]
    fn a_non_unicode_mmproj_path_is_refused_before_the_model_loads() {
        with_phase_setup(NON_GGUF_MODEL, |phase_setup| {
            let projector_setup = ProjectorSetup {
                mmproj_source: LOCAL_MMPROJ,
                mmproj_file: OnceLock::from(MmprojFile::from_path(PathBuf::from(
                    OsString::from_vec(vec![0xff]),
                ))),
                mtmd_context: OnceLock::new(),
                mtmd_context_poisoning: OnceLock::new(),
                phase_setup,
            };

            assert_eq!(
                projector_setup
                    .load_projector_state()
                    .err()
                    .map(PhaseSetupError::kind),
                Some(PhaseSetupFailureKind::MmprojPathNotUnicode)
            );
        });
    }

    #[test]
    fn a_projector_a_trial_panicked_while_holding_fails_every_later_trial_as_poisoned() {
        let projector = Mutex::new(());
        let poisoning = OnceLock::new();

        assert!(
            thread::scope(|scope| {
                scope
                    .spawn(|| {
                        let _held_projector = projector.lock();
                        let _second_projector_lock = projector
                            .try_lock()
                            .expect("the trial panics while it holds the projector");
                    })
                    .join()
            })
            .is_err()
        );

        let poisoned_failure = lock_projector(&projector, &poisoning, LOCAL_MMPROJ)
            .expect_err("a poisoned projector must fail the trial that locks it");
        let later_failure = lock_projector(&projector, &poisoning, LOCAL_MMPROJ)
            .expect_err("a poisoned projector must fail every later trial too");

        assert!(matches!(
            poisoned_failure,
            PhaseSetupError::MmprojPoisoned { mmproj_source } if *mmproj_source == LOCAL_MMPROJ
        ));
        assert!(ptr::eq(poisoned_failure, later_failure));
    }

    #[test]
    fn a_model_that_cannot_be_loaded_fails_the_projector_with_the_cached_model_load() {
        with_phase_setup(NON_GGUF_MODEL, |phase_setup| {
            let projector_setup = ProjectorSetup::new(LOCAL_MMPROJ, Arc::clone(&phase_setup));

            let model_load_failure = phase_setup
                .load_phase_state()
                .expect_err("a model file that is not GGUF must fail to load");
            let projector_failure = projector_setup
                .load_projector_state()
                .expect_err("a projector whose model failed to load must fail to set up");

            assert!(ptr::eq(model_load_failure, projector_failure));
        });
    }
}
