use std::ffi::CStr;
use std::ffi::c_char;
use std::str::Utf8Error;

/// The id mtmd reports for a bitmap or a chunk, or `None` when mtmd reports no id.
///
/// # Errors
///
/// Returns [`Utf8Error`] when the id is not valid UTF-8.
///
/// # Safety
///
/// `id` must be null or point to a NUL-terminated string that outlives the call.
pub unsafe fn optional_mtmd_id(id: *const c_char) -> Result<Option<String>, Utf8Error> {
    if id.is_null() {
        return Ok(None);
    }

    Ok(Some(unsafe { CStr::from_ptr(id) }.to_str()?.to_owned()))
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::optional_mtmd_id;

    #[test]
    fn a_missing_id_is_none() {
        assert_eq!(unsafe { optional_mtmd_id(ptr::null()) }, Ok(None));
    }

    #[test]
    fn an_id_is_copied() {
        assert_eq!(
            unsafe { optional_mtmd_id(c"image".as_ptr()) },
            Ok(Some("image".to_owned()))
        );
    }

    #[test]
    fn an_id_that_is_not_utf8_is_refused() {
        let not_utf8 = c"\xff";

        assert_eq!(
            unsafe { optional_mtmd_id(not_utf8.as_ptr()) },
            Err(not_utf8.to_str().expect_err("0xff is not UTF-8"))
        );
    }
}
