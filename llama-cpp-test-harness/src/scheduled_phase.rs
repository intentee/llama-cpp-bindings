use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use libtest_mimic::Trial;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_test_model_provisioning::phase_setup::PhaseSetup;

use crate::execution_phase::ExecutionPhase;
use crate::llama_fixture::LlamaFixture;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::trial_outcome::TrialOutcome;

fn trial_outcome<'setup>(
    registration: &'static LlamaTestRegistration,
    setup: &'setup PhaseSetup,
) -> TrialOutcome<'setup> {
    match setup.load_phase_state() {
        Ok(state) => match registration.func.run(&LlamaFixture {
            model: &state.model,
            backend: &state.backend,
            context_params: &registration.context_params,
            mtmd_context: state.mtmd_context.as_ref(),
            model_path: &state.model_path,
        }) {
            Ok(()) => TrialOutcome::Succeeded,
            Err(test_failure) => TrialOutcome::TestFailed(test_failure),
        },
        Err(setup_failure) => TrialOutcome::PhaseSetupFailed(setup_failure),
    }
}

fn trial(registration: &'static LlamaTestRegistration, setup: &Arc<PhaseSetup>) -> Trial {
    let setup = Arc::clone(setup);
    let expected_outcome = registration.expected_outcome;

    expected_outcome.label(Trial::test(registration.name, move || {
        expected_outcome.judge(trial_outcome(registration, &setup))
    }))
}

pub struct ScheduledPhase<'plan> {
    pub phase: &'plan ExecutionPhase,
    pub setup: Arc<PhaseSetup>,
    pub trials: Vec<Trial>,
}

impl<'plan> ScheduledPhase<'plan> {
    #[must_use]
    pub fn new(phase: &'plan ExecutionPhase, backend: &Arc<LlamaBackend>) -> Self {
        let setup = Arc::new(PhaseSetup::new(phase.key, Arc::clone(backend)));
        let trials = phase
            .registrations
            .iter()
            .map(|registration| trial(registration, &setup))
            .collect();

        Self {
            phase,
            setup,
            trials,
        }
    }

    #[must_use]
    pub fn selects_any_trial(&self, arguments: &Arguments) -> bool {
        !arguments.list
            && self
                .trials
                .iter()
                .any(|trial| !arguments.is_filtered_out(trial) && !arguments.is_ignored(trial))
    }

    pub fn download_model_files(&self, arguments: &Arguments) {
        if self.selects_any_trial(arguments)
            && let Err(download_failure) = self.setup.download_model_files()
        {
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
}

#[cfg(test)]
mod tests {
    use std::iter;
    use std::sync::Arc;
    use std::sync::PoisonError;

    use libtest_mimic::Arguments;

    use llama_cpp_bindings::llama_backend::LlamaBackend;
    use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;
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
    use crate::no_op::no_op;

    const LOAD_KEY: LoadKey = LoadKey {
        model_source: GgufSource::LocalPath("/nonexistent/llama-cpp-test-harness/model.gguf"),
        mmproj_source: None,
        model_load_params: ModelLoadParams {
            n_gpu_layers: 0,
            load_mode: LlamaLoadMode::Mmap,
        },
    };

    const CONTEXT_PARAMS: ContextParams = ContextParams {
        n_ctx: 1,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    };

    static REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
        name: "scheduled_trial",
        key: LOAD_KEY,
        context_params: CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Success,
    };

    static FAILING_ON_PURPOSE_REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
        name: "failing_on_purpose",
        key: LOAD_KEY,
        context_params: CONTEXT_PARAMS,
        func: LlamaTestFn::Infallible(no_op),
        expected_outcome: ExpectedOutcome::Failure {
            failure: ExpectedFailure::PhaseSetup(PhaseSetupFailureKind::ModelLoad),
            reason: "the model file does not exist on purpose",
        },
    };

    fn with_scheduled_phase<TResult, TUse: FnOnce(&ScheduledPhase<'_>) -> TResult>(
        registrations: Vec<&'static LlamaTestRegistration>,
        use_scheduled_phase: TUse,
    ) -> TResult {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let phase = ExecutionPhase {
            key: LOAD_KEY,
            registrations,
        };

        use_scheduled_phase(&ScheduledPhase::new(&phase, &backend))
    }

    fn selects_any_trial_with(harness_arguments: &[&str]) -> bool {
        let arguments =
            Arguments::from_iter(iter::once("harness").chain(harness_arguments.iter().copied()));

        with_scheduled_phase(vec![&REGISTRATION], |scheduled_phase| {
            scheduled_phase.selects_any_trial(&arguments)
        })
    }

    #[test]
    fn only_a_trial_declared_to_fail_is_labelled_with_the_failure_it_expects_and_why() {
        assert_eq!(
            with_scheduled_phase(
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
}
