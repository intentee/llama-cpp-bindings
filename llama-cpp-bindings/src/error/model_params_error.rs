#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ModelParamsError {
    #[error("No available slot in override vector")]
    NoAvailableSlot,
    #[error("Override slot is not empty")]
    SlotNotEmpty,
    #[error("the key needs {length_with_nul} bytes with its NUL, but llama.cpp holds {capacity}")]
    KeyTooLong {
        length_with_nul: usize,
        capacity: usize,
    },
}
