use crate::error::unknown_pooling_type::UnknownPoolingType;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaPoolingType {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl TryFrom<llama_cpp_bindings_sys::llama_pooling_type> for LlamaPoolingType {
    type Error = UnknownPoolingType;

    fn try_from(value: llama_cpp_bindings_sys::llama_pooling_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_UNSPECIFIED => Ok(Self::Unspecified),
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_MEAN => Ok(Self::Mean),
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_CLS => Ok(Self::Cls),
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_LAST => Ok(Self::Last),
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_RANK => Ok(Self::Rank),
            unknown => Err(UnknownPoolingType(unknown)),
        }
    }
}

impl From<LlamaPoolingType> for llama_cpp_bindings_sys::llama_pooling_type {
    fn from(value: LlamaPoolingType) -> Self {
        match value {
            LlamaPoolingType::Unspecified => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_UNSPECIFIED,
            LlamaPoolingType::None => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_NONE,
            LlamaPoolingType::Mean => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_MEAN,
            LlamaPoolingType::Cls => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_CLS,
            LlamaPoolingType::Last => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_LAST,
            LlamaPoolingType::Rank => llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_RANK,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaPoolingType;
    use crate::error::unknown_pooling_type::UnknownPoolingType;

    #[test]
    fn every_llama_cpp_value_converts_to_its_variant_and_back() {
        let raw_values = [
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_UNSPECIFIED,
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_NONE,
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_MEAN,
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_CLS,
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_LAST,
            llama_cpp_bindings_sys::LLAMA_POOLING_TYPE_RANK,
        ];

        assert_eq!(
            raw_values.map(LlamaPoolingType::try_from),
            [
                Ok(LlamaPoolingType::Unspecified),
                Ok(LlamaPoolingType::None),
                Ok(LlamaPoolingType::Mean),
                Ok(LlamaPoolingType::Cls),
                Ok(LlamaPoolingType::Last),
                Ok(LlamaPoolingType::Rank),
            ]
        );
        assert_eq!(
            [
                LlamaPoolingType::Unspecified,
                LlamaPoolingType::None,
                LlamaPoolingType::Mean,
                LlamaPoolingType::Cls,
                LlamaPoolingType::Last,
                LlamaPoolingType::Rank,
            ]
            .map(llama_cpp_bindings_sys::llama_pooling_type::from),
            raw_values
        );
    }

    #[test]
    fn a_value_llama_cpp_does_not_define_is_refused() {
        assert_eq!(LlamaPoolingType::try_from(99), Err(UnknownPoolingType(99)));
    }
}
