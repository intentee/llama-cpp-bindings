#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod context_params;
pub mod download_model;
pub mod execution_phase;
pub mod execution_plan;
pub mod gguf_source;
pub mod harness_arguments_error;
pub mod harness_run_error;
pub mod llama_fixture;
pub mod llama_test_fn;
pub mod llama_test_registration;
pub mod llama_tests_main_macro;
pub mod load_key;
pub mod model_load_params;
pub mod no_op;
pub mod parse_harness_arguments;
pub mod phase_model_files;
pub mod phase_setup;
pub mod phase_state;
pub mod run;
pub mod run_to_conclusions;
pub mod scheduled_phase;
#[cfg(test)]
mod test_backend_gate;

#[doc(hidden)]
pub use inventory;

pub use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;
pub use llama_cpp_test_harness_macros::llama_test;

pub use crate::context_params::ContextParams;
pub use crate::execution_phase::ExecutionPhase;
pub use crate::execution_plan::ExecutionPlan;
pub use crate::gguf_source::GgufSource;
pub use crate::llama_fixture::LlamaFixture;
pub use crate::llama_test_fn::LlamaTestFn;
pub use crate::llama_test_registration::LlamaTestRegistration;
pub use crate::load_key::LoadKey;
pub use crate::model_load_params::ModelLoadParams;
pub use crate::no_op::no_op;
pub use crate::phase_model_files::PhaseModelFiles;
pub use crate::phase_setup::PhaseSetup;
pub use crate::phase_state::PhaseState;
pub use crate::run::run;
pub use crate::run_to_conclusions::run_to_conclusions;
pub use crate::scheduled_phase::ScheduledPhase;
