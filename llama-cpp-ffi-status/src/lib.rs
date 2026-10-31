#![cfg_attr(
    not(test),
    deny(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )
)]

pub mod ffi_contract_error;
pub mod ffi_status_error;
pub mod read_and_free_cpp_string;
