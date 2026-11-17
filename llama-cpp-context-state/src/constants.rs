use crate::llama_state_seq_flags::LlamaStateSeqFlags;

pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: LlamaStateSeqFlags = LlamaStateSeqFlags { flags: 1 };

#[cfg(test)]
mod tests {
    use super::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    #[test]
    fn partial_only_has_bit_one() {
        assert_eq!(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY.bits(), 1);
    }
}
