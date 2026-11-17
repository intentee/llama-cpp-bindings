use llama_cpp_mtmd::mtmd_context::MtmdContext;

use crate::phase_state::PhaseState;

#[derive(Debug)]
pub struct ProjectorState<'setup> {
    pub phase_state: &'setup PhaseState,
    pub mtmd_context: &'setup MtmdContext,
}
