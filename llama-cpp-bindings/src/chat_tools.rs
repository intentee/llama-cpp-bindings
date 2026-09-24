use std::ffi::CStr;
use std::ffi::CString;

use crate::error::ChatToolsError;

#[derive(Debug)]
pub struct ChatTools {
    json: CString,
}

impl ChatTools {
    /// # Errors
    /// Returns [`ChatToolsError`] when `json` contains a NUL byte, is not valid JSON, or is not
    /// a JSON array.
    pub fn from_json(json: String) -> Result<Self, ChatToolsError> {
        let json = CString::new(json).map_err(ChatToolsError::ContainsNulByte)?;
        let json_value: serde_json::Value =
            serde_json::from_slice(json.as_bytes()).map_err(ChatToolsError::InvalidJson)?;

        if !json_value.is_array() {
            return Err(ChatToolsError::NotArray);
        }

        Ok(Self { json })
    }

    #[must_use]
    pub fn json_cstr(&self) -> &CStr {
        &self.json
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::ChatTools;

    #[test]
    fn keeps_a_valid_tools_array_for_parsing() {
        let tools = ChatTools::from_json("[]".to_owned()).unwrap();

        assert_eq!(tools.json_cstr().to_bytes(), b"[]");
    }

    #[test]
    fn rejects_malformed_json() {
        let json_error = serde_json::from_str::<Value>("not_a_json[}").unwrap_err();

        assert_eq!(
            ChatTools::from_json("not_a_json[}".to_owned())
                .unwrap_err()
                .to_string(),
            format!("chat tools are not valid JSON: {json_error}")
        );
    }

    #[test]
    fn rejects_json_that_is_not_an_array() {
        assert_eq!(
            ChatTools::from_json("{\"foo\": 1}".to_owned())
                .unwrap_err()
                .to_string(),
            "chat tools must be a JSON array"
        );
    }

    #[test]
    fn reports_a_nul_byte_followed_by_extra_text_as_a_nul_byte() {
        assert_eq!(
            ChatTools::from_json("[]\0extra".to_owned())
                .unwrap_err()
                .to_string(),
            "chat tools contain an interior NUL byte at position 2"
        );
    }
}
