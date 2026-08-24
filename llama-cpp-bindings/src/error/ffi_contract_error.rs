#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{operation} returned an invalid FFI response: {detail}")]
pub struct FfiContractError {
    pub operation: &'static str,
    pub detail: &'static str,
}
