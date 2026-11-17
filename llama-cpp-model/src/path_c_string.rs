use std::ffi::CString;
use std::path::Path;

use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

use crate::model_error::ModelError;

/// # Errors
///
/// Returns [`ModelError::PathNotUtf8`] when `path` is not valid UTF-8, or
/// [`ModelError::PathContainsNulByte`] when it contains a NUL byte.
pub fn path_c_string(path: &Path) -> Result<CString, ModelError> {
    llama_cpp_path_c_string::path_c_string::path_c_string(path).map_err(|error| match error {
        PathCStringError::ContainsNulByte(source) => ModelError::PathContainsNulByte(source),
        PathCStringError::NotUtf8(path) => ModelError::PathNotUtf8(path),
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::path_c_string;
    use crate::model_error::ModelError;

    #[test]
    fn a_utf8_path_becomes_its_c_string() {
        assert_eq!(
            path_c_string(Path::new("/models/decision.gguf")),
            Ok(c"/models/decision.gguf".to_owned())
        );
    }

    #[test]
    fn a_path_that_is_not_utf8_is_refused() {
        let path = Path::new(OsStr::from_bytes(b"/models/\xff\xfe.gguf"));

        assert_eq!(
            path_c_string(path),
            Err(ModelError::PathNotUtf8(path.to_path_buf()))
        );
    }

    #[test]
    fn a_path_with_a_nul_byte_is_refused() {
        assert_eq!(
            path_c_string(Path::new("/models/deci\0sion.gguf")),
            Err(ModelError::PathContainsNulByte(
                CString::new("/models/deci\0sion.gguf").unwrap_err()
            ))
        );
    }
}
