use std::ffi::CString;

use crate::error::new_llama_chat_message_error::NewLlamaChatMessageError;

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct LlamaChatMessage {
    pub role: CString,
    pub content: CString,
}

impl LlamaChatMessage {
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
    use std::ffi::CString;

    use crate::error::new_llama_chat_message_error::NewLlamaChatMessageError;
    use crate::model::llama_chat_message::LlamaChatMessage;

    #[test]
    fn a_role_and_content_without_nul_bytes_form_a_message() {
        assert_eq!(
            LlamaChatMessage::new("user".to_string(), "hello".to_string()),
            Ok(LlamaChatMessage {
                role: c"user".to_owned(),
                content: c"hello".to_owned(),
            })
        );
    }

    #[test]
    fn a_nul_byte_in_the_role_is_reported() {
        assert_eq!(
            LlamaChatMessage::new("us\0er".to_string(), "hello".to_string()),
            Err(NewLlamaChatMessageError::NulError(
                CString::new("us\0er").unwrap_err()
            ))
        );
    }

    #[test]
    fn a_nul_byte_in_the_content_is_reported() {
        assert_eq!(
            LlamaChatMessage::new("user".to_string(), "hel\0lo".to_string()),
            Err(NewLlamaChatMessageError::NulError(
                CString::new("hel\0lo").unwrap_err()
            ))
        );
    }
}
