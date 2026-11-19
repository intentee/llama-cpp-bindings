use enumflags2::bitflags;

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaStateSeqFlag {
    PartialOnly = 1 << 0,
    OnDevice = 1 << 1,
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;

    use super::LlamaStateSeqFlag;

    #[test]
    fn each_flag_is_the_bit_llama_cpp_defines_for_it() {
        assert_eq!(
            [LlamaStateSeqFlag::PartialOnly, LlamaStateSeqFlag::OnDevice]
                .map(|flag| BitFlags::from_flag(flag).bits()),
            [
                llama_cpp_bindings_sys::bindings::LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY,
                llama_cpp_bindings_sys::bindings::LLAMA_STATE_SEQ_FLAGS_ON_DEVICE,
            ]
        );
    }
}
