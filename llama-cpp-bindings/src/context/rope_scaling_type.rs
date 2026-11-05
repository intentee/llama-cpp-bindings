use crate::error::unknown_rope_scaling_type::UnknownRopeScalingType;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeScalingType {
    Unspecified,
    None,
    Linear,
    Yarn,
    LongRope,
}

impl TryFrom<llama_cpp_bindings_sys::llama_rope_scaling_type> for RopeScalingType {
    type Error = UnknownRopeScalingType;

    fn try_from(
        value: llama_cpp_bindings_sys::llama_rope_scaling_type,
    ) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED => Ok(Self::Unspecified),
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LINEAR => Ok(Self::Linear),
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_YARN => Ok(Self::Yarn),
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LONGROPE => Ok(Self::LongRope),
            unknown => Err(UnknownRopeScalingType(unknown)),
        }
    }
}

impl From<RopeScalingType> for llama_cpp_bindings_sys::llama_rope_scaling_type {
    fn from(value: RopeScalingType) -> Self {
        match value {
            RopeScalingType::Unspecified => {
                llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
            }
            RopeScalingType::None => llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_NONE,
            RopeScalingType::Linear => llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LINEAR,
            RopeScalingType::Yarn => llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_YARN,
            RopeScalingType::LongRope => llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LONGROPE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RopeScalingType;
    use crate::error::unknown_rope_scaling_type::UnknownRopeScalingType;

    #[test]
    fn every_llama_cpp_value_converts_to_its_variant_and_back() {
        let raw_values = [
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_NONE,
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LINEAR,
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_YARN,
            llama_cpp_bindings_sys::LLAMA_ROPE_SCALING_TYPE_LONGROPE,
        ];

        assert_eq!(
            raw_values.map(RopeScalingType::try_from),
            [
                Ok(RopeScalingType::Unspecified),
                Ok(RopeScalingType::None),
                Ok(RopeScalingType::Linear),
                Ok(RopeScalingType::Yarn),
                Ok(RopeScalingType::LongRope),
            ]
        );
        assert_eq!(
            [
                RopeScalingType::Unspecified,
                RopeScalingType::None,
                RopeScalingType::Linear,
                RopeScalingType::Yarn,
                RopeScalingType::LongRope,
            ]
            .map(llama_cpp_bindings_sys::llama_rope_scaling_type::from),
            raw_values
        );
    }

    #[test]
    fn a_value_llama_cpp_does_not_define_is_refused() {
        assert_eq!(
            RopeScalingType::try_from(99),
            Err(UnknownRopeScalingType(99))
        );
    }
}
