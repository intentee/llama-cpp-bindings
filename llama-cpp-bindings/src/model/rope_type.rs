use crate::error::unknown_rope_type::UnknownRopeType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    None,
    Norm,
    NeoX,
    MRope,
    IMRope,
    Vision,
}

impl TryFrom<llama_cpp_bindings_sys::llama_rope_type> for RopeType {
    type Error = UnknownRopeType;

    fn try_from(value: llama_cpp_bindings_sys::llama_rope_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NONE => Ok(Self::None),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM => Ok(Self::Norm),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX => Ok(Self::NeoX),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE => Ok(Self::MRope),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE => Ok(Self::IMRope),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION => Ok(Self::Vision),
            unknown => Err(UnknownRopeType(unknown)),
        }
    }
}

impl RopeType {
    /// Whether every embedding carries a single position, which llama.cpp requires before it
    /// shifts or divides the positions of a sequence in memory.
    #[must_use]
    pub const fn has_one_position_per_embedding(self) -> bool {
        match self {
            Self::None | Self::Norm | Self::NeoX | Self::Vision => true,
            Self::MRope | Self::IMRope => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::error::unknown_rope_type::UnknownRopeType;
    use crate::model::rope_type::RopeType;

    #[test]
    fn every_llama_cpp_rope_type_converts_to_its_variant() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NONE,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION,
            ]
            .map(RopeType::try_from),
            [
                Ok(RopeType::None),
                Ok(RopeType::Norm),
                Ok(RopeType::NeoX),
                Ok(RopeType::MRope),
                Ok(RopeType::IMRope),
                Ok(RopeType::Vision),
            ]
        );
    }

    #[test]
    fn a_value_llama_cpp_does_not_define_is_refused() {
        assert_eq!(RopeType::try_from(9999), Err(UnknownRopeType(9999)));
    }

    #[test]
    fn only_multi_section_rope_types_place_several_positions_per_embedding() {
        assert_eq!(
            [
                RopeType::None,
                RopeType::Norm,
                RopeType::NeoX,
                RopeType::MRope,
                RopeType::IMRope,
                RopeType::Vision,
            ]
            .map(RopeType::has_one_position_per_embedding),
            [true, true, true, false, false, true]
        );
    }
}
