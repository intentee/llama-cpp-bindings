use std::sync::MutexGuard;

use llama_cpp_mtmd::mtmd_context::MtmdContext;

use crate::phase_state::PhaseState;

/// The model a phase keeps loaded, together with the projector a trial declares, which stays
/// locked for as long as the trial keeps this state, so no other trial uses it at the same time.
#[derive(Debug)]
pub struct ProjectorState<'setup> {
    pub phase_state: &'setup PhaseState,
    pub mtmd_context: MutexGuard<'setup, MtmdContext>,
}
