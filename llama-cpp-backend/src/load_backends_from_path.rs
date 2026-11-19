use std::path::Path;

use llama_cpp_path_c_string::path_c_string::path_c_string;

use crate::backend_error::BackendError;

/// # Errors
///
/// Returns [`BackendError::BackendDirectoryPathInvalid`] when `path` is not valid UTF-8 or
/// contains an interior null byte.
pub fn load_backends_from_path(path: &Path) -> Result<(), BackendError> {
    let path_cstring = path_c_string(path).map_err(BackendError::BackendDirectoryPathInvalid)?;

    unsafe {
        llama_cpp_bindings_sys::bindings::ggml_backend_load_all_from_path(path_cstring.as_ptr());
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

    use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

    use super::load_backends_from_path;
    use crate::backend_error::BackendError;

    #[test]
    fn load_backends_from_path_returns_path_null_byte_for_embedded_null() {
        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/foo\0bar"));

        assert_eq!(
            load_backends_from_path(&path),
            Err(BackendError::BackendDirectoryPathInvalid(
                PathCStringError::ContainsNulByte(CString::new("/tmp/foo\0bar").unwrap_err())
            ))
        );
    }

    #[test]
    fn load_backends_from_path_returns_path_not_utf8_for_invalid_utf8() {
        let path = PathBuf::from(OsStr::from_bytes(b"/tmp/\xff\xfe"));

        assert_eq!(
            load_backends_from_path(&path),
            Err(BackendError::BackendDirectoryPathInvalid(
                PathCStringError::NotUtf8(path.clone())
            ))
        );
    }

    #[test]
    fn a_directory_without_backend_libraries_loads_no_backend() {
        let registered_before =
            unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_reg_count() };

        assert_eq!(load_backends_from_path(&env::temp_dir()), Ok(()));
        assert_eq!(
            unsafe { llama_cpp_bindings_sys::bindings::ggml_backend_reg_count() },
            registered_before
        );
    }
}
