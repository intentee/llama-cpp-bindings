/// a rusty equivalent of `llama_vocab_type`
#[repr(u32)]
#[derive(Debug, Eq, Copy, Clone, PartialEq)]
pub enum VocabType {
    /// Byte Pair Encoding
    BPE = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE as _,
    /// Sentence Piece Tokenizer
    SPM = llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM as _,
}

/// There was an error converting a `llama_vocab_type` to a `VocabType`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`. Contains the int value that was invalid.
    #[error("Unknown Value {0}")]
    UnknownValue(llama_cpp_bindings_sys::llama_vocab_type),
}

impl TryFrom<llama_cpp_bindings_sys::llama_vocab_type> for VocabType {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_bindings_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE => Ok(VocabType::BPE),
            llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM => Ok(VocabType::SPM),
            unknown => Err(LlamaTokenTypeFromIntError::UnknownValue(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaTokenTypeFromIntError, VocabType};

    #[test]
    fn try_from_bpe() {
        let result = VocabType::try_from(llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_BPE);

        assert_eq!(result, Ok(VocabType::BPE));
    }

    #[test]
    fn try_from_spm() {
        let result = VocabType::try_from(llama_cpp_bindings_sys::LLAMA_VOCAB_TYPE_SPM);

        assert_eq!(result, Ok(VocabType::SPM));
    }

    #[test]
    fn try_from_unknown_value() {
        let result = VocabType::try_from(99999);

        assert_eq!(result, Err(LlamaTokenTypeFromIntError::UnknownValue(99999)));
    }
}
