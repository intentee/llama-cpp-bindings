use crate::error::unknown_attention_type::UnknownAttentionType;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaAttentionType {
    Unspecified,
    Causal,
    NonCausal,
}

impl TryFrom<llama_cpp_bindings_sys::llama_attention_type> for LlamaAttentionType {
    type Error = UnknownAttentionType;

    fn try_from(value: llama_cpp_bindings_sys::llama_attention_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_UNSPECIFIED => Ok(Self::Unspecified),
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_CAUSAL => Ok(Self::Causal),
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_NON_CAUSAL => Ok(Self::NonCausal),
            unknown => Err(UnknownAttentionType(unknown)),
        }
    }
}

impl From<LlamaAttentionType> for llama_cpp_bindings_sys::llama_attention_type {
    fn from(value: LlamaAttentionType) -> Self {
        match value {
            LlamaAttentionType::Unspecified => {
                llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_UNSPECIFIED
            }
            LlamaAttentionType::Causal => llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_CAUSAL,
            LlamaAttentionType::NonCausal => {
                llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_NON_CAUSAL
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaAttentionType;
    use crate::error::unknown_attention_type::UnknownAttentionType;

    #[test]
    fn every_llama_cpp_value_converts_to_its_variant_and_back() {
        let raw_values = [
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_UNSPECIFIED,
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_CAUSAL,
            llama_cpp_bindings_sys::LLAMA_ATTENTION_TYPE_NON_CAUSAL,
        ];

        assert_eq!(
            raw_values.map(LlamaAttentionType::try_from),
            [
                Ok(LlamaAttentionType::Unspecified),
                Ok(LlamaAttentionType::Causal),
                Ok(LlamaAttentionType::NonCausal),
            ]
        );
        assert_eq!(
            [
                LlamaAttentionType::Unspecified,
                LlamaAttentionType::Causal,
                LlamaAttentionType::NonCausal,
            ]
            .map(llama_cpp_bindings_sys::llama_attention_type::from),
            raw_values
        );
    }

    #[test]
    fn a_value_llama_cpp_does_not_define_is_refused() {
        assert_eq!(
            LlamaAttentionType::try_from(99),
            Err(UnknownAttentionType(99))
        );
    }
}
