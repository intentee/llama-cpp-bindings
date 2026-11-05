#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod backend_init_gate;
pub mod download_model;
pub mod gguf_source;
pub mod load_key;
pub mod model_load_params;
pub mod phase_model_files;
pub mod phase_setup;
pub mod phase_setup_error;
pub mod phase_setup_failure_kind;
pub mod phase_state;
