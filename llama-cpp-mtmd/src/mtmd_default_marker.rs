use std::ffi::CStr;
use std::ffi::c_char;

use crate::mtmd_error::MtmdError;

unsafe fn marker_bytes_to_str(c_str: *const c_char) -> Result<&'static str, MtmdError> {
    unsafe { CStr::from_ptr(c_str) }
        .to_str()
        .map_err(MtmdError::DefaultMarkerNotUtf8)
}

/// # Errors
///
/// Returns [`MtmdError::DefaultMarkerNotUtf8`] if llama.cpp's `mtmd_default_marker`
/// returns bytes that are not valid UTF-8. The marker is a fixed ASCII constant in
/// the llama.cpp library; surfacing the error keeps the failure explicit rather than
/// papering over it with a substituted literal.
pub fn mtmd_default_marker() -> Result<&'static str, MtmdError> {
    unsafe { marker_bytes_to_str(llama_cpp_bindings_sys::bindings::mtmd_default_marker()) }
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;
    use std::str;

    use super::marker_bytes_to_str;
    use crate::mtmd_default_marker::mtmd_default_marker;
    use crate::mtmd_error::MtmdError;

    #[test]
    fn the_default_marker_is_the_media_marker() {
        assert_eq!(mtmd_default_marker(), Ok("<__media__>"));
    }

    #[test]
    fn non_utf8_marker_bytes_return_not_utf8_error() {
        let invalid: [u8; 3] = [0xFF, 0xFE, 0x00];
        let result = unsafe { marker_bytes_to_str(invalid.as_ptr().cast::<c_char>()) };

        assert_eq!(
            result.err(),
            str::from_utf8(&invalid[..2])
                .err()
                .map(MtmdError::DefaultMarkerNotUtf8)
        );
    }
}
