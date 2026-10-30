#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecodedOutputRows {
    BatchIndices(Vec<i32>),
    LastRow,
}

impl DecodedOutputRows {
    /// The rows a llama.cpp helper writes when it decodes on its own batches, which output at most
    /// the last row and only when asked to.
    #[must_use]
    pub const fn of_helper_decode(logits_last: bool) -> Self {
        if logits_last {
            Self::LastRow
        } else {
            Self::BatchIndices(Vec::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DecodedOutputRows;

    #[test]
    fn a_helper_decode_with_logits_last_outputs_its_last_row() {
        assert_eq!(
            DecodedOutputRows::of_helper_decode(true),
            DecodedOutputRows::LastRow
        );
    }

    #[test]
    fn a_helper_decode_without_logits_last_outputs_no_row() {
        assert_eq!(
            DecodedOutputRows::of_helper_decode(false),
            DecodedOutputRows::BatchIndices(Vec::new())
        );
    }
}
