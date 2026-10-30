use crate::model::vocab_type_from_int_error::VocabTypeFromIntError;

#[repr(u32)]
#[derive(Debug, Eq, Copy, Clone, PartialEq)]
pub enum VocabType {
    None = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_NONE as _,
    Spm = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM as _,
    Bpe = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE as _,
    Wpm = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_WPM as _,
    Ugm = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_UGM as _,
    Rwkv = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_RWKV as _,
    Plamo2 = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO2 as _,
    Test = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_TEST as _,
    Plamo3 = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO3 as _,
}

impl TryFrom<llama_cpp_bindings_sys::llama_vocab_type> for VocabType {
    type Error = VocabTypeFromIntError;

    fn try_from(value: llama_cpp_bindings_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM => Ok(Self::Spm),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE => Ok(Self::Bpe),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_WPM => Ok(Self::Wpm),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_UGM => Ok(Self::Ugm),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_RWKV => Ok(Self::Rwkv),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO2 => Ok(Self::Plamo2),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_TEST => Ok(Self::Test),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO3 => Ok(Self::Plamo3),
            unknown => Err(VocabTypeFromIntError::UnknownValue(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{VocabType, VocabTypeFromIntError};

    #[test]
    fn every_llama_cpp_vocab_type_maps_to_its_variant() {
        let mapped = [
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_NONE,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_WPM,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_UGM,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_RWKV,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO2,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_TEST,
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_PLAMO3,
        ]
        .map(VocabType::try_from);

        assert_eq!(
            mapped,
            [
                Ok(VocabType::None),
                Ok(VocabType::Spm),
                Ok(VocabType::Bpe),
                Ok(VocabType::Wpm),
                Ok(VocabType::Ugm),
                Ok(VocabType::Rwkv),
                Ok(VocabType::Plamo2),
                Ok(VocabType::Test),
                Ok(VocabType::Plamo3),
            ]
        );
    }

    #[test]
    fn a_vocab_type_unknown_to_llama_cpp_is_refused() {
        let result = VocabType::try_from(99999);

        assert_eq!(result, Err(VocabTypeFromIntError::UnknownValue(99999)));
    }
}
