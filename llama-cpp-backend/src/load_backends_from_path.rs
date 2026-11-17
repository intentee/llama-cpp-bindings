use std::ffi::CString;
use std::path::Path;

use crate::backend_error::BackendError;

/// # Errors
///
/// Returns [`BackendError::BackendDirectoryPathNotUtf8`] when `path` cannot be converted to UTF-8
/// and [`BackendError::BackendDirectoryPathContainsNullByte`] when the path contains an interior
/// null byte.
pub fn load_backends_from_path(path: &Path) -> Result<(), BackendError> {
    let path_str = path
        .to_str()
        .ok_or_else(|| BackendError::BackendDirectoryPathNotUtf8(path.to_path_buf()))?;
    let path_cstring = CString::new(path_str)?;

    unsafe {
        llama_cpp_bindings_sys::ggml_backend_load_all_from_path(path_cstring.as_ptr());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    use std::path::PathBuf;

    use std::env;

    use super::load_backends_from_path;
    use crate::backend_error::BackendError;

    #[test]
    fn load_backends_from_path_returns_path_null_byte_for_embedded_null() {
        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/foo\0bar"));

        assert_eq!(
            load_backends_from_path(&path),
            Err(BackendError::BackendDirectoryPathContainsNullByte(
                CString::new("/tmp/foo\0bar").unwrap_err()
            ))
        );
    }

    #[test]
    fn load_backends_from_path_returns_path_not_utf8_for_invalid_utf8() {
        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/\xff\xfe"));

        assert_eq!(
            load_backends_from_path(&path),
            Err(BackendError::BackendDirectoryPathNotUtf8(path.clone()))
        );
    }

    #[test]
    fn a_directory_without_backend_libraries_loads_no_backend() {
        let registered_before = unsafe { llama_cpp_bindings_sys::ggml_backend_reg_count() };

        assert_eq!(load_backends_from_path(&env::temp_dir()), Ok(()));
        assert_eq!(
            unsafe { llama_cpp_bindings_sys::ggml_backend_reg_count() },
            registered_before
        );
    }
}
