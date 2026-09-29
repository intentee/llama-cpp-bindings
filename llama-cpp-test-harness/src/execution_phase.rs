use std::sync::Arc;
use std::sync::OnceLock;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;
use libtest_mimic::Failed;
use libtest_mimic::Trial;
use llama_cpp_bindings::llama_backend::LlamaBackend;

use crate::GgufSource;
use crate::llama_fixture::LlamaFixture;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::load_key::LoadKey;
use crate::phase_state::PhaseState;

type LazyPhaseState = Arc<OnceLock<Result<PhaseState, String>>>;

fn source_label(source: GgufSource) -> String {
    match source {
        GgufSource::HuggingFace { repo, file } => format!("{repo} / {file}"),
        GgufSource::LocalPath(path) => format!("local:{path}"),
    }
}

pub struct ExecutionPhase {
    pub key: LoadKey,
    pub registrations: Vec<&'static LlamaTestRegistration>,
}

impl ExecutionPhase {
    #[must_use]
    pub fn header_line(&self, index: usize, total: usize) -> String {
        format!(
            "--- phase {phase_number}/{total_phases}: {source_label} (n_gpu_layers={n_gpu_layers}) ({trial_count} tests) ---",
            phase_number = index + 1,
            total_phases = total,
            source_label = source_label(self.key.model_source),
            n_gpu_layers = self.key.model_load_params.n_gpu_layers,
            trial_count = self.registrations.len(),
        )
    }

    pub fn print_header(&self, index: usize, total: usize) {
        eprintln!("{}", self.header_line(index, total));
    }

    pub fn run(&self, backend: &Arc<LlamaBackend>, arguments: &Arguments) -> Conclusion {
        let phase_state: LazyPhaseState = Arc::new(OnceLock::new());

        libtest_mimic::run(
            arguments,
            self.registrations
                .iter()
                .map(|registration| self.trial(registration, backend, &phase_state))
                .collect(),
        )
    }

    fn trial(
        &self,
        registration: &'static LlamaTestRegistration,
        backend: &Arc<LlamaBackend>,
        phase_state: &LazyPhaseState,
    ) -> Trial {
        let key = self.key;
        let backend = Arc::clone(backend);
        let phase_state = Arc::clone(phase_state);

        Trial::test(registration.name, move || {
            match phase_state.get_or_init(|| {
                key.load_phase_state(&backend)
                    .map_err(|error| format!("phase setup failed: {error:#}"))
            }) {
                Ok(state) => (registration.func)(&LlamaFixture {
                    model: &state.model,
                    backend: &state.backend,
                    context_params: &registration.context_params,
                    mtmd_context: state.mtmd_context.as_ref(),
                    model_path: &state.model_path,
                })
                .map_err(|error| Failed::from(format!("{error:#}"))),
                Err(setup_failure) => Err(Failed::from(setup_failure)),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::GgufSource;
    use crate::LlamaLoadMode;
    use crate::load_key::LoadKey;
    use crate::model_load_params::ModelLoadParams;

    use super::ExecutionPhase;

    fn phase_with_source(source: GgufSource) -> ExecutionPhase {
        ExecutionPhase {
            key: LoadKey {
                model_source: source,
                mmproj_source: None,
                model_load_params: ModelLoadParams {
                    n_gpu_layers: 7,
                    load_mode: LlamaLoadMode::Mmap,
                },
            },
            registrations: Vec::new(),
        }
    }

    #[test]
    fn header_line_for_huggingface_source_formats_repo_and_file() {
        let phase = phase_with_source(GgufSource::HuggingFace {
            repo: "org/name",
            file: "model.gguf",
        });

        let line = phase.header_line(0, 4);

        assert_eq!(
            line,
            "--- phase 1/4: org/name / model.gguf (n_gpu_layers=7) (0 tests) ---"
        );
    }

    #[test]
    fn header_line_for_local_path_source_uses_local_prefix() {
        let phase = phase_with_source(GgufSource::LocalPath("/abs/model.gguf"));

        let line = phase.header_line(2, 3);

        assert_eq!(
            line,
            "--- phase 3/3: local:/abs/model.gguf (n_gpu_layers=7) (0 tests) ---"
        );
    }
}
