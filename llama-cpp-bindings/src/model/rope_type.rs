#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Norm,
    NeoX,
    MRope,
    IMRope,
    Vision,
}

impl RopeType {
    #[must_use]
    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM => Some(Self::Norm),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX => Some(Self::NeoX),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE => Some(Self::MRope),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE => Some(Self::IMRope),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION => Some(Self::Vision),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::model::rope_type::RopeType;

    #[test]
    fn llama_cpp_rope_types_map_to_their_variants_and_the_rest_to_none() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_IMROPE,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION,
                llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NONE,
                9999,
            ]
            .map(RopeType::from_raw),
            [
                Some(RopeType::Norm),
                Some(RopeType::NeoX),
                Some(RopeType::MRope),
                Some(RopeType::IMRope),
                Some(RopeType::Vision),
                None,
                None,
            ]
        );
    }
}
