use std::ffi::CStr;
use std::ffi::CString;

use crate::error::ChatToolsError;

#[derive(Debug)]
pub struct ChatTools {
    json: String,
    json_cstring: CString,
}

impl ChatTools {
    /// # Errors
    /// Returns [`ChatToolsError`] when `json` contains a NUL byte, is not valid JSON, or is not
    /// a JSON array.
    pub fn from_json(json: String) -> Result<Self, ChatToolsError> {
        let json_cstring =
            CString::new(json.as_bytes()).map_err(ChatToolsError::ContainsNulByte)?;
        let json_value: serde_json::Value =
            serde_json::from_str(&json).map_err(ChatToolsError::InvalidJson)?;

        if !json_value.is_array() {
            return Err(ChatToolsError::NotArray);
        }

        Ok(Self { json, json_cstring })
    }

    #[must_use]
    pub fn json(&self) -> &str {
        &self.json
    }

    #[must_use]
    pub fn json_cstr(&self) -> &CStr {
        &self.json_cstring
    }
}

#[cfg(test)]
mod tests {
    use super::ChatTools;
    use crate::error::ChatToolsError;

    #[test]
    fn keeps_a_valid_tools_array_for_parsing() {
        let tools = ChatTools::from_json("[]".to_owned()).unwrap();

        assert_eq!(tools.json(), "[]");
        assert_eq!(tools.json_cstr().to_bytes(), b"[]");
    }

    #[test]
    fn rejects_malformed_json() {
        assert!(matches!(
            ChatTools::from_json("not_a_json[}".to_owned()),
            Err(ChatToolsError::InvalidJson(_))
        ));
    }

    #[test]
    fn rejects_json_that_is_not_an_array() {
        assert!(matches!(
            ChatTools::from_json("{\"foo\": 1}".to_owned()),
            Err(ChatToolsError::NotArray)
        ));
    }

    #[test]
    fn reports_a_nul_byte_followed_by_extra_text_as_a_nul_byte() {
        assert!(matches!(
            ChatTools::from_json("[]\0extra".to_owned()),
            Err(ChatToolsError::ContainsNulByte(nul_error)) if nul_error.nul_position() == 2
        ));
    }

    #[test]
    fn reports_a_trailing_nul_byte_as_a_nul_byte() {
        assert!(matches!(
            ChatTools::from_json("[]\0".to_owned()),
            Err(ChatToolsError::ContainsNulByte(nul_error)) if nul_error.nul_position() == 2
        ));
    }
}
