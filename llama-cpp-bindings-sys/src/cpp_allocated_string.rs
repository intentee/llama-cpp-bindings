use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr::NonNull;
use std::string::FromUtf8Error;

use crate::bindings::llama_rs_string_free;
use crate::exception_message_error::ExceptionMessageError;

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

    /// Reads this string as the message of a C++ exception the `operation` reported, building
    /// the error `reported` makes of it, or the error type's own error for a message that is not
    /// valid UTF-8.
    pub fn into_reported_error<TError>(
        self,
        operation: &'static str,
        reported: impl FnOnce(String) -> TError,
    ) -> TError
    where
        TError: ExceptionMessageError,
    {
        self.into_string().map_or_else(
            |source| TError::exception_message_not_utf8(operation, source),
            reported,
        )
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
    use crate::exception_message_error::ExceptionMessageError;

    #[derive(Debug, PartialEq, Eq)]
    enum ReportedFailure {
        Reported(String),
        ExceptionMessageNotUtf8 {
            operation: &'static str,
            source: FromUtf8Error,
        },
    }

    impl ExceptionMessageError for ReportedFailure {
        fn exception_message_not_utf8(operation: &'static str, source: FromUtf8Error) -> Self {
            Self::ExceptionMessageNotUtf8 { operation, source }
        }
    }

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

    #[test]
    fn into_reported_error_builds_the_reported_error_from_the_message() {
        let message = unsafe {
            CppAllocatedString::from_raw(llama_rs_string_dup(c"llama.cpp threw".as_ptr()))
        };

        assert_eq!(
            message.into_reported_error("llama_rs_process", ReportedFailure::Reported),
            ReportedFailure::Reported("llama.cpp threw".to_owned())
        );
    }

    #[test]
    fn into_reported_error_reports_a_message_that_is_not_utf8() {
        let message =
            unsafe { CppAllocatedString::from_raw(llama_rs_string_dup(c"\xff".as_ptr())) };

        assert_eq!(
            message.into_reported_error("llama_rs_process", ReportedFailure::Reported),
            ReportedFailure::ExceptionMessageNotUtf8 {
                operation: "llama_rs_process",
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            }
        );
    }
}
