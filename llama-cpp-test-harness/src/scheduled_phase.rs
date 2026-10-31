use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use libtest_mimic::Failed;
use libtest_mimic::Trial;

use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::execution_phase::ExecutionPhase;
use crate::llama_fixture::LlamaFixture;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::phase_setup::PhaseSetup;

fn trial(registration: &'static LlamaTestRegistration, setup: &Arc<PhaseSetup>) -> Trial {
    let setup = Arc::clone(setup);

    Trial::test(registration.name, move || match setup.load_phase_state() {
        Ok(state) => registration
            .func
            .run(&LlamaFixture {
                model: &state.model,
                backend: &state.backend,
                context_params: &registration.context_params,
                mtmd_context: state.mtmd_context.as_ref(),
                model_path: &state.model_path,
            })
            .map_err(|error| Failed::from(format!("{error:#}"))),
        Err(setup_failure) => Err(Failed::from(format!(
            "phase setup failed: {setup_failure:#}"
        ))),
    })
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

    use super::ScheduledPhase;
    use crate::context_params::ContextParams;
    use crate::execution_phase::ExecutionPhase;
    use crate::gguf_source::GgufSource;
    use crate::llama_test_fn::LlamaTestFn;
    use crate::llama_test_registration::LlamaTestRegistration;
    use crate::load_key::LoadKey;
    use crate::model_load_params::ModelLoadParams;
    use crate::no_op::no_op;
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    static REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
        name: "scheduled_trial",
        key: LoadKey {
            model_source: GgufSource::LocalPath("/nonexistent/llama-cpp-test-harness/model.gguf"),
            mmproj_source: None,
            model_load_params: ModelLoadParams {
                n_gpu_layers: 0,
                load_mode: LlamaLoadMode::Mmap,
            },
        },
        context_params: ContextParams {
            n_ctx: 1,
            n_batch: 1,
            n_ubatch: 1,
            n_seq_max: 1,
            n_threads_batch: None,
            embeddings: false,
        },
        void_logs: false,
        func: LlamaTestFn::Infallible(no_op),
    };

    fn selects_any_trial_with(harness_arguments: &[&str]) -> bool {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let backend = Arc::new(LlamaBackend::init().expect("backend init must succeed"));
        let phase = ExecutionPhase {
            key: REGISTRATION.key,
            registrations: vec![&REGISTRATION],
        };
        let arguments =
            Arguments::from_iter(iter::once("harness").chain(harness_arguments.iter().copied()));

        ScheduledPhase::new(&phase, &backend).selects_any_trial(&arguments)
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
