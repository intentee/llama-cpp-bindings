use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr::NonNull;
use std::string::FromUtf8Error;

use crate::bindings::llama_rs_string_free;

/// A string the C++ wrappers allocated and handed over through an out-parameter, freed with
/// `llama_rs_string_free` when dropped.
#[derive(Debug)]
pub struct CppAllocatedString {
    pointer: NonNull<c_char>,
}

impl CppAllocatedString {
    /// # Safety
    ///
    /// `pointer` must be a non-null, NUL-terminated string allocated by `llama_rs_dup_string`
    /// that nothing else frees.
    #[must_use]
    pub const unsafe fn from_raw(pointer: *mut c_char) -> Self {
        Self {
            pointer: unsafe { NonNull::new_unchecked(pointer) },
        }
    }

    /// # Errors
    ///
    /// Returns [`FromUtf8Error`], keeping the bytes, when the string is not valid UTF-8.
    pub fn into_string(self) -> Result<String, FromUtf8Error> {
        String::from_utf8(
            unsafe { CStr::from_ptr(self.pointer.as_ptr()) }
                .to_bytes()
                .to_vec(),
        )
    }
}

impl Drop for CppAllocatedString {
    fn drop(&mut self) {
        unsafe { llama_rs_string_free(self.pointer.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use std::string::FromUtf8Error;

    use super::CppAllocatedString;
    use crate::bindings::llama_rs_string_dup;

    #[test]
    fn into_string_reads_the_text_the_cpp_side_allocated() {
        let string =
            unsafe { CppAllocatedString::from_raw(llama_rs_string_dup(c"llama".as_ptr())) };

        assert_eq!(string.into_string(), Ok("llama".to_owned()));
    }

    #[test]
    fn into_string_keeps_the_bytes_of_text_that_is_not_utf8() {
        let string =
            unsafe { CppAllocatedString::from_raw(llama_rs_string_dup(c"llama\xff".as_ptr())) };

        assert_eq!(
            string.into_string().map_err(FromUtf8Error::into_bytes),
            Err(b"llama\xff".to_vec())
        );
    }
}
