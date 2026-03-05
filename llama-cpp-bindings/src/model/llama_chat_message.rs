use std::ffi::CString;

use crate::NewLlamaChatMessageError;

/// A Safe wrapper around `llama_chat_message`
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    pub(super) role: CString,
    pub(super) content: CString,
}

impl LlamaChatMessage {
    /// Create a new `LlamaChatMessage`
    ///
    /// # Errors
    /// If either of ``role`` or ``content`` contain null bytes.
    pub fn new(role: String, content: String) -> Result<Self, NewLlamaChatMessageError> {
        Ok(Self {
            role: CString::new(role)?,
            content: CString::new(content)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaChatMessage;

    #[test]
    fn valid_construction() {
        let message = LlamaChatMessage::new("user".to_string(), "hello".to_string());

        assert!(message.is_ok());
    }

    #[test]
    fn null_byte_in_role_returns_error() {
        let message = LlamaChatMessage::new("us\0er".to_string(), "hello".to_string());

        assert!(message.is_err());
    }

    #[test]
    fn null_byte_in_content_returns_error() {
        let message = LlamaChatMessage::new("user".to_string(), "hel\0lo".to_string());

        assert!(message.is_err());
    }

    #[test]
    fn empty_strings_are_valid() {
        let message = LlamaChatMessage::new(String::new(), String::new());

        assert!(message.is_ok());
    }
}
