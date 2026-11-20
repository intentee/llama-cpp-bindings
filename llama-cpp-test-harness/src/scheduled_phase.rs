use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use libtest_mimic::Trial;

use llama_cpp_backend::llama_backend::LlamaBackend;
use llama_cpp_test_model_provisioning::gguf_source::GgufSource;
use llama_cpp_test_model_provisioning::phase_setup::PhaseSetup;
use llama_cpp_test_model_provisioning::phase_setup_error::PhaseSetupError;
use llama_cpp_test_model_provisioning::projector_setup::ProjectorSetup;

use crate::execution_phase::ExecutionPhase;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::trial_setup::TrialSetup;

fn trial(registration: &'static LlamaTestRegistration, trial_setup: TrialSetup) -> Trial {
    let expected_outcome = registration.expected_outcome;

    expected_outcome.label(Trial::test(registration.name, move || {
        expected_outcome.judge(trial_setup.trial_outcome(registration))
    }))
}

fn selects(arguments: &Arguments, trial: &Trial) -> bool {
    !arguments.list && !arguments.is_filtered_out(trial) && !arguments.is_ignored(trial)
}

pub struct ScheduledPhase<'plan> {
    pub phase: &'plan ExecutionPhase,
    pub setup: Arc<PhaseSetup>,
    pub projector_setups: BTreeMap<GgufSource, Arc<ProjectorSetup>>,
    pub trials: Vec<Trial>,
}

impl<'plan> ScheduledPhase<'plan> {
    #[must_use]
    pub fn new(phase: &'plan ExecutionPhase, backend: &Arc<LlamaBackend>) -> Self {
        let setup = Arc::new(PhaseSetup::new(phase.key, Arc::clone(backend)));
        let mut projector_setups = BTreeMap::new();
        let trials = phase
            .registrations
            .iter()
            .map(|registration| {
                let trial_setup = registration.key.mmproj_source.map_or_else(
                    || TrialSetup::ModelOnly(Arc::clone(&setup)),
                    |mmproj_source| {
                        TrialSetup::WithProjector(Arc::clone(
                            projector_setups.entry(mmproj_source).or_insert_with(|| {
                                Arc::new(ProjectorSetup::new(mmproj_source, Arc::clone(&setup)))
                            }),
                        ))
                    },
                );

                trial(registration, trial_setup)
            })
            .collect();

        Self {
            phase,
            setup,
            projector_setups,
            trials,
        }
    }

    #[must_use]
    pub fn download_failures(&self, arguments: &Arguments) -> Vec<&PhaseSetupError> {
        if !self.selects_any_trial(arguments) {
            return Vec::new();
        }

        match self.setup.download_model_file() {
            Ok(_) => self.mmproj_download_failures(arguments),
            Err(download_failure) => vec![download_failure],
        }
    }

    pub fn download_model_files(&self, arguments: &Arguments) {
        for download_failure in self.download_failures(arguments) {
            eprintln!("model files download failed: {download_failure:#}");
        }
    }

    pub fn run(self, index: usize, total: usize, arguments: &Arguments) -> Conclusion {
        self.phase.print_header(index, total);

        if self.selects_any_trial(arguments)
            && let Err(setup_failure) = self.setup.load_phase_state()
        {
            eprintln!("phase setup failed: {setup_failure:#}");
        }

        libtest_mimic::run(arguments, self.trials)
    }

    #[must_use]
    pub fn selected_mmproj_sources(&self, arguments: &Arguments) -> BTreeSet<GgufSource> {
        self.phase
            .registrations
            .iter()
            .zip(&self.trials)
            .filter(|(_, trial)| selects(arguments, trial))
            .filter_map(|(registration, _)| registration.key.mmproj_source)
            .collect()
    }

    #[must_use]
    pub fn selects_any_trial(&self, arguments: &Arguments) -> bool {
        self.trials.iter().any(|trial| selects(arguments, trial))
    }

    fn mmproj_download_failures(&self, arguments: &Arguments) -> Vec<&PhaseSetupError> {
        let selected_mmproj_sources = self.selected_mmproj_sources(arguments);

        self.projector_setups
            .iter()
            .filter(|(mmproj_source, _)| selected_mmproj_sources.contains(mmproj_source))
            .filter_map(|(_, projector_setup)| projector_setup.download_mmproj_file().err())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::iter;
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::PoisonError;

    use libtest_mimic::Arguments;
    use libtest_mimic::Conclusion;

    use llama_cpp_backend::llama_backend::LlamaBackend;
    use llama_cpp_params::llama_load_mode::LlamaLoadMode;
    use llama_cpp_test_model_provisioning::backend_init_gate::BACKEND_INIT_GATE;
    use llama_cpp_test_model_provisioning::gguf_source::GgufSource;
    use llama_cpp_test_model_provisioning::load_key::LoadKey;
    use llama_cpp_test_model_provisioning::model_load_params::ModelLoadParams;
    use llama_cpp_test_model_provisioning::phase_setup_failure_kind::PhaseSetupFailureKind;

    use super::ScheduledPhase;
    use crate::context_params::ContextParams;
    use crate::execution_phase::ExecutionPhase;
    use crate::expected_failure::ExpectedFailure;
    use crate::expected_outcome::ExpectedOutcome;
    use crate::llama_test_fn::LlamaTestFn;
    use crate::llama_test_registration::LlamaTestRegistration;

    const MODEL_LOAD_PARAMS: ModelLoadParams = ModelLoadParams {
        n_gpu_layers: 0,
        load_mode: LlamaLoadMode::Mmap,
    };

    const NON_GGUF_MODEL: GgufSource =
        GgufSource::LocalPath(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"));

    const MISSING_MODEL: GgufSource = GgufSource::HuggingFace {
        repo: "intentee-test-harness/does-not-exist",
        file: "no-such-file.gguf",
    };

    const LOCAL_MMPROJ: GgufSource =
        GgufSource::LocalPath("/nonexistent/llama-cpp-test-harness/mmproj.gguf");

    const MISSING_MMPROJ: GgufSource = GgufSource::HuggingFace {
        repo: "intentee-test-harness/does-not-exist",
        file: "no-such-mmproj.gguf",
    };

    const SKIPPED_MMPROJ: GgufSource = GgufSource::HuggingFace {
        repo: "intentee-test-harness/does-not-exist",
        file: "no-such-skipped-mmproj.gguf",
    };

    const LOAD_KEY: LoadKey = LoadKey {
        model_source: NON_GGUF_MODEL,
        mmproj_source: None,
        model_load_params: MODEL_LOAD_PARAMS,
    };

    const CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 1,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    fn registration_expecting(
        name: &'static str,
        model_source: GgufSource,
        mmproj_source: Option<GgufSource>,
        expected_kind: PhaseSetupFailureKind,
    ) -> LlamaTestRegistration {
        LlamaTestRegistration {
            name,
            key: LoadKey {
                model_source,
                mmproj_source,
                model_load_params: MODEL_LOAD_PARAMS,
            },
            context_params: CONTEXT_PARAMS,
            func: LlamaTestFn::no_op(),
            expected_outcome: ExpectedOutcome::Failure {
                failure: ExpectedFailure::PhaseSetup(expected_kind),
                reason: "the setup fails on purpose",
            },
        }
    }

    static REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
        name: "scheduled_trial",
        key: LOAD_KEY,
        context_params: CONTEXT_PARAMS,
        func: LlamaTestFn::no_op(),
        expected_outcome: ExpectedOutcome::Success,
    };

    static FAILING_ON_PURPOSE_REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
        name: "failing_on_purpose",
        key: LOAD_KEY,
        context_params: CONTEXT_PARAMS,
        func: LlamaTestFn::no_op(),
        expected_outcome: ExpectedOutcome::Failure {
            failure: ExpectedFailure::PhaseSetup(PhaseSetupFailureKind::ModelLoad),
            reason: "the model file does not exist on purpose",
        },
    };

    static WITHOUT_MMPROJ_ON_A_NON_GGUF_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "without_mmproj_on_a_non_gguf_model",
                NON_GGUF_MODEL,
                None,
                PhaseSetupFailureKind::ModelLoad,
            )
        });

    static LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "local_mmproj_on_a_non_gguf_model",
                NON_GGUF_MODEL,
                Some(LOCAL_MMPROJ),
                PhaseSetupFailureKind::ModelLoad,
            )
        });

    static SECOND_LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "second_local_mmproj_on_a_non_gguf_model",
                NON_GGUF_MODEL,
                Some(LOCAL_MMPROJ),
                PhaseSetupFailureKind::ModelLoad,
            )
        });

    static MISSING_MMPROJ_ON_A_NON_GGUF_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "missing_mmproj_on_a_non_gguf_model",
                NON_GGUF_MODEL,
                Some(MISSING_MMPROJ),
                PhaseSetupFailureKind::MmprojDownload,
            )
        });

    static SKIPPED_MMPROJ_ON_A_NON_GGUF_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "skipped_mmproj_on_a_non_gguf_model",
                NON_GGUF_MODEL,
                Some(SKIPPED_MMPROJ),
                PhaseSetupFailureKind::MmprojDownload,
            )
        });

    static WITHOUT_MMPROJ_ON_A_MISSING_MODEL: LazyLock<LlamaTestRegistration> =
        LazyLock::new(|| {
            registration_expecting(
                "without_mmproj_on_a_missing_model",
                MISSING_MODEL,
                None,
                PhaseSetupFailureKind::ModelDownload,
            )
        });

    static LOCAL_MMPROJ_ON_A_MISSING_MODEL: LazyLock<LlamaTestRegistration> = LazyLock::new(|| {
        registration_expecting(
            "local_mmproj_on_a_missing_model",
            MISSING_MODEL,
            Some(LOCAL_MMPROJ),
            PhaseSetupFailureKind::ModelDownload,
        )
    });

    fn arguments_with(harness_arguments: &[&str]) -> Arguments {
        Arguments::from_iter(iter::once("harness").chain(harness_arguments.iter().copied()))
    }

    fn with_scheduled_phase<TResult, TUse: FnOnce(ScheduledPhase<'_>) -> TResult>(
        model_source: GgufSource,
        registrations: Vec<&'static LlamaTestRegistration>,
        use_scheduled_phase: TUse,
    ) -> TResult {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let phase = ExecutionPhase {
            key: LoadKey {
                model_source,
                mmproj_source: None,
                model_load_params: MODEL_LOAD_PARAMS,
            }
            .model_key(),
            registrations,
        };

        use_scheduled_phase(ScheduledPhase::new(&phase, &backend))
    }

    fn selects_any_trial_with(harness_arguments: &[&str]) -> bool {
        let arguments = arguments_with(harness_arguments);

        with_scheduled_phase(NON_GGUF_MODEL, vec![&REGISTRATION], |scheduled_phase| {
            scheduled_phase.selects_any_trial(&arguments)
        })
    }

    fn conclusion_of_downloading_then_running(
        model_source: GgufSource,
        registrations: Vec<&'static LlamaTestRegistration>,
        harness_arguments: &[&str],
    ) -> Conclusion {
        let arguments = arguments_with(harness_arguments);

        with_scheduled_phase(model_source, registrations, |scheduled_phase| {
            scheduled_phase.download_model_files(&arguments);

            scheduled_phase.run(0, 1, &arguments)
        })
    }

    #[test]
    fn only_a_trial_declared_to_fail_is_labelled_with_the_failure_it_expects_and_why() {
        assert_eq!(
            with_scheduled_phase(
                NON_GGUF_MODEL,
                vec![&REGISTRATION, &FAILING_ON_PURPOSE_REGISTRATION],
                |scheduled_phase| {
                    scheduled_phase
                        .trials
                        .iter()
                        .map(|trial| trial.kind().to_owned())
                        .collect::<Vec<String>>()
                }
            ),
            [
                String::new(),
                "expected ModelLoad: the model file does not exist on purpose".to_owned()
            ]
        );
    }

    #[test]
    fn selects_any_trial_when_a_trial_matches_the_filter() {
        assert!(selects_any_trial_with(&["scheduled"]));
    }

    #[test]
    fn selects_no_trial_when_every_trial_is_filtered_out() {
        assert!(!selects_any_trial_with(&["--exact", "another_trial"]));
    }

    #[test]
    fn selects_no_trial_when_listing() {
        assert!(!selects_any_trial_with(&["--list"]));
    }

    #[test]
    fn a_phase_whose_trials_are_all_filtered_out_attempts_no_download() {
        let arguments = arguments_with(&["--exact", "another_trial"]);

        assert_eq!(
            with_scheduled_phase(
                MISSING_MODEL,
                vec![&WITHOUT_MMPROJ_ON_A_MISSING_MODEL],
                |scheduled_phase| scheduled_phase.download_failures(&arguments).len()
            ),
            0
        );
    }

    #[test]
    fn selects_only_the_mmproj_sources_of_trials_that_are_not_filtered_out() {
        let arguments = arguments_with(&["--skip", "skipped_mmproj"]);

        assert_eq!(
            with_scheduled_phase(
                NON_GGUF_MODEL,
                vec![
                    &WITHOUT_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &SKIPPED_MMPROJ_ON_A_NON_GGUF_MODEL,
                ],
                |scheduled_phase| scheduled_phase.selected_mmproj_sources(&arguments)
            ),
            BTreeSet::from([LOCAL_MMPROJ])
        );
    }

    #[test]
    fn trials_sharing_an_mmproj_source_share_one_projector_setup() {
        assert_eq!(
            with_scheduled_phase(
                NON_GGUF_MODEL,
                vec![
                    &WITHOUT_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &SECOND_LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &MISSING_MMPROJ_ON_A_NON_GGUF_MODEL,
                ],
                |scheduled_phase| {
                    scheduled_phase
                        .projector_setups
                        .keys()
                        .copied()
                        .collect::<Vec<GgufSource>>()
                }
            ),
            [MISSING_MMPROJ, LOCAL_MMPROJ]
        );
    }

    #[test]
    fn a_model_that_cannot_be_downloaded_fails_every_trial_of_its_phase() {
        assert_eq!(
            conclusion_of_downloading_then_running(
                MISSING_MODEL,
                vec![
                    &WITHOUT_MMPROJ_ON_A_MISSING_MODEL,
                    &LOCAL_MMPROJ_ON_A_MISSING_MODEL,
                ],
                &["--test-threads=1"],
            ),
            Conclusion {
                num_filtered_out: 0,
                num_passed: 2,
                num_failed: 0,
                num_ignored: 0,
                num_measured: 0,
            }
        );
    }

    #[test]
    fn each_trial_fails_with_the_setup_failure_of_the_part_it_depends_on() {
        assert_eq!(
            conclusion_of_downloading_then_running(
                NON_GGUF_MODEL,
                vec![
                    &WITHOUT_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &LOCAL_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &MISSING_MMPROJ_ON_A_NON_GGUF_MODEL,
                    &SKIPPED_MMPROJ_ON_A_NON_GGUF_MODEL,
                ],
                &["--test-threads=1", "--skip", "skipped_mmproj"],
            ),
            Conclusion {
                num_filtered_out: 1,
                num_passed: 3,
                num_failed: 0,
                num_ignored: 0,
                num_measured: 0,
            }
        );
    }
}
