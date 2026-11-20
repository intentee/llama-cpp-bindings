use std::ffi::CString;
use std::path::Path;

use crate::path_c_string_error::PathCStringError;

/// # Errors
///
/// Returns [`PathCStringError::NotUtf8`] when `path` is not valid UTF-8, or
/// [`PathCStringError::ContainsNulByte`] when it contains a NUL byte.
pub fn path_c_string(path: &Path) -> Result<CString, PathCStringError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| PathCStringError::NotUtf8(path.to_path_buf()))?;

    CString::new(path_str).map_err(PathCStringError::ContainsNulByte)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::path::Path;

    use super::path_c_string;
    use crate::path_c_string_error::PathCStringError;

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
            Err(PathCStringError::NotUtf8(path.to_path_buf()))
        );
    }

    #[test]
    fn a_path_with_a_nul_byte_is_refused() {
        assert_eq!(
            path_c_string(Path::new("/models/deci\0sion.gguf")),
            Err(PathCStringError::ContainsNulByte(
                CString::new("/models/deci\0sion.gguf").unwrap_err()
            ))
        );
    }
}
