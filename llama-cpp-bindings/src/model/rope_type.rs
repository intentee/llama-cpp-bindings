#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    Norm,
    NeoX,
    MRope,
    Vision,
}

impl RopeType {
    #[must_use]
    pub const fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM => Some(Self::Norm),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX => Some(Self::NeoX),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE => Some(Self::MRope),
            llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION => Some(Self::Vision),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RopeType;

    #[test]
    fn rope_type_none() {
        assert_eq!(
            RopeType::from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NONE),
            None
        );
    }

    #[test]
    fn rope_type_norm() {
        assert_eq!(
            RopeType::from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NORM),
            Some(RopeType::Norm)
        );
    }

    #[test]
    fn rope_type_neox() {
        assert_eq!(
            RopeType::from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_NEOX),
            Some(RopeType::NeoX)
        );
    }

    #[test]
    fn rope_type_mrope() {
        assert_eq!(
            RopeType::from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_MROPE),
            Some(RopeType::MRope)
        );
    }

    #[test]
    fn rope_type_vision() {
        assert_eq!(
            RopeType::from_raw(llama_cpp_bindings_sys::LLAMA_ROPE_TYPE_VISION),
            Some(RopeType::Vision)
        );
    }

    #[test]
    fn rope_type_unknown_returns_none() {
        assert_eq!(RopeType::from_raw(9999), None);
    }
}
