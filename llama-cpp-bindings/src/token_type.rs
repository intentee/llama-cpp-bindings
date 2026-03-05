//! Utilities for working with `llama_token_type` values.
use enumflags2::{BitFlags, bitflags};
use std::ops::{Deref, DerefMut};

/// A rust flavored equivalent of `llama_token_type`.
#[derive(Eq, PartialEq, Debug, Clone, Copy)]
#[bitflags]
#[repr(u32)]
pub enum LlamaTokenAttr {
    /// Unknown token attribute.
    Unknown = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNKNOWN as _,
    /// Unused token attribute.
    Unused = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_UNUSED as _,
    /// Normal text token.
    Normal = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMAL as _,
    /// Control token (e.g. BOS, EOS).
    Control = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_CONTROL as _,
    /// User-defined token.
    UserDefined = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_USER_DEFINED as _,
    /// Byte-level fallback token.
    Byte = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_BYTE as _,
    /// Token with normalized text.
    Normalized = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMALIZED as _,
    /// Token with left-stripped whitespace.
    LStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_LSTRIP as _,
    /// Token with right-stripped whitespace.
    RStrip = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_RSTRIP as _,
    /// Token representing a single word.
    SingleWord = llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_SINGLE_WORD as _,
}

/// A set of `LlamaTokenAttrs`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaTokenAttrs(pub BitFlags<LlamaTokenAttr>);

impl Deref for LlamaTokenAttrs {
    type Target = BitFlags<LlamaTokenAttr>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LlamaTokenAttrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TryFrom<llama_cpp_bindings_sys::llama_token_type> for LlamaTokenAttrs {
    type Error = LlamaTokenTypeFromIntError;

    fn try_from(value: llama_cpp_bindings_sys::llama_vocab_type) -> Result<Self, Self::Error> {
        Ok(Self(BitFlags::from_bits(value as _).map_err(|e| {
            LlamaTokenTypeFromIntError::UnknownValue(e.invalid_bits())
        })?))
    }
}

/// An error type for `LlamaTokenType::try_from`.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum LlamaTokenTypeFromIntError {
    /// The value is not a valid `llama_token_type`.
    #[error("Unknown Value {0}")]
    UnknownValue(std::ffi::c_uint),
}

#[cfg(test)]
mod tests {
    use enumflags2::BitFlags;

    use super::{LlamaTokenAttr, LlamaTokenAttrs, LlamaTokenTypeFromIntError};

    #[test]
    fn try_from_valid_single_attribute() {
        let attrs =
            LlamaTokenAttrs::try_from(llama_cpp_bindings_sys::LLAMA_TOKEN_ATTR_NORMAL as u32);

        assert!(attrs.is_ok());
        assert!(
            attrs
                .expect("valid attribute")
                .contains(LlamaTokenAttr::Normal)
        );
    }

    #[test]
    fn try_from_zero_produces_empty_flags() {
        let attrs = LlamaTokenAttrs::try_from(0u32);

        assert!(attrs.is_ok());
        assert!(attrs.expect("valid attribute").is_empty());
    }

    #[test]
    fn try_from_invalid_bits_returns_error() {
        let invalid_value = 0xFFFF_FFFFu32;
        let result = LlamaTokenAttrs::try_from(invalid_value);

        assert!(result.is_err());
        matches!(
            result.expect_err("should fail"),
            LlamaTokenTypeFromIntError::UnknownValue(_)
        );
    }

    #[test]
    fn deref_exposes_bitflags_methods() {
        let attrs = LlamaTokenAttrs(BitFlags::from_flag(LlamaTokenAttr::Control));

        assert!(attrs.contains(LlamaTokenAttr::Control));
        assert!(!attrs.contains(LlamaTokenAttr::Normal));
    }

    #[test]
    fn deref_mut_allows_modification() {
        let mut attrs = LlamaTokenAttrs(BitFlags::empty());

        attrs.insert(LlamaTokenAttr::Byte);

        assert!(attrs.contains(LlamaTokenAttr::Byte));
    }
}
