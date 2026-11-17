use std::sync::Arc;

use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_test_model_provisioning::phase_setup::PhaseSetup;
use llama_cpp_test_model_provisioning::phase_state::PhaseState;
use llama_cpp_test_model_provisioning::projector_setup::ProjectorSetup;
use llama_cpp_test_model_provisioning::projector_state::ProjectorState;

use crate::llama_fixture::LlamaFixture;
use crate::llama_test_registration::LlamaTestRegistration;
use crate::trial_outcome::TrialOutcome;

fn test_outcome<'setup>(
    LlamaTestRegistration {
        func,
        context_params,
        ..
    }: &'static LlamaTestRegistration,
    PhaseState {
        model,
        backend,
        model_path,
    }: &PhaseState,
    mtmd_context: Option<&MtmdContext>,
) -> TrialOutcome<'setup> {
    match func.run(&LlamaFixture {
        model,
        backend,
        context_params,
        mtmd_context,
        model_path,
    }) {
        Ok(()) => TrialOutcome::Succeeded,
        Err(test_failure) => TrialOutcome::TestFailed(test_failure),
    }
}

/// What a trial runs against: the model its phase keeps loaded, alone or together with the
/// multimodal projector the trial declares.
pub enum TrialSetup {
    ModelOnly(Arc<PhaseSetup>),
    WithProjector(Arc<ProjectorSetup>),
}

impl TrialSetup {
    #[must_use]
    pub fn trial_outcome(&self, registration: &'static LlamaTestRegistration) -> TrialOutcome<'_> {
        match self {
            Self::ModelOnly(phase_setup) => match phase_setup.load_phase_state() {
                Ok(phase_state) => test_outcome(registration, phase_state, None),
                Err(setup_failure) => TrialOutcome::PhaseSetupFailed(setup_failure),
            },
            Self::WithProjector(projector_setup) => match projector_setup.load_projector_state() {
                Ok(ProjectorState {
                    phase_state,
                    mtmd_context,
                }) => test_outcome(registration, phase_state, Some(mtmd_context)),
                Err(setup_failure) => TrialOutcome::PhaseSetupFailed(setup_failure),
            },
        }
    }
}
