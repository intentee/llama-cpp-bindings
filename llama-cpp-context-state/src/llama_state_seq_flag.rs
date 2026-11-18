use enumflags2::bitflags;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaStateSeqFlag {
    PartialOnly = 1 << 0,
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;

    use super::LlamaStateSeqFlag;

    #[test]
    fn partial_only_is_the_bit_llama_cpp_defines_for_partial_states() {
        assert_eq!(
            BitFlags::from_flag(LlamaStateSeqFlag::PartialOnly).bits(),
            1
        );
    }
}
