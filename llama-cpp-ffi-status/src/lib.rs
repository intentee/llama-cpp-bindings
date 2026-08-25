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

pub use ffi_contract_error::FfiContractError;
pub use ffi_status_error::FfiStatusError;
pub use read_and_free_cpp_string::read_and_free_cpp_string;
