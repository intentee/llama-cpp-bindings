#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )
)]

pub mod compile_command;
pub mod compile_commands_file;
pub mod cpp_standard;
pub mod wrapper_headers;
pub mod wrapper_include_dirs;
pub mod wrapper_source_paths;
pub mod wrapper_sources;
pub mod wrapper_sources_error;
pub mod wrapper_sources_response_file;
