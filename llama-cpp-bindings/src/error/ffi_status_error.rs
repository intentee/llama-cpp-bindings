#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("{operation} returned unknown FFI status {code}")]
pub struct FfiStatusError {
    pub operation: &'static str,
    pub code: u32,
}
