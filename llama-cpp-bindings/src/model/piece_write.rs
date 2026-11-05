use std::ffi::c_int;

/// What `llama_token_to_piece` did with the buffer it was given: wrote the piece's bytes, or
/// reported, as a negative size, the bytes the piece needs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceWrite {
    Written(usize),
    NeedsBuffer(c_int),
}

impl From<c_int> for PieceWrite {
    fn from(size: c_int) -> Self {
        if size.is_negative() {
            Self::NeedsBuffer(size)
        } else {
            Self::Written(size.unsigned_abs() as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::PieceWrite;

    #[test]
    fn a_negative_size_asks_for_a_larger_buffer_and_any_other_is_the_written_length() {
        assert_eq!(
            [-7, 0, 5].map(PieceWrite::from),
            [
                PieceWrite::NeedsBuffer(-7),
                PieceWrite::Written(0),
                PieceWrite::Written(5)
            ]
        );
    }
}
