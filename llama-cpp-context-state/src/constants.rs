use crate::llama_state_seq_flags::LlamaStateSeqFlags;

pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: LlamaStateSeqFlags = LlamaStateSeqFlags::from_bits(1);

#[cfg(test)]
mod tests {
    use super::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
    use crate::llama_state_seq_flags::LlamaStateSeqFlags;

    #[test]
    fn partial_only_has_bit_one() {
        assert_eq!(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY.bits(), 1);
    }

    #[test]
    fn empty_does_not_contain_partial_only() {
        let flags = LlamaStateSeqFlags::empty();

        assert!(!flags.contains(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    }
}
