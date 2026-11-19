use std::ffi::c_char;

use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::chat_error::ChatError;

/// Builds the error of a status that carries the message of a thrown C++ exception, or
/// [`ChatError::ExceptionMessageNotUtf8`] when that message is not valid UTF-8.
///
/// # Safety
///
/// `out_error` must be the non-null exception message the C++ side allocated for a status that
/// carries one; it is freed here.
pub unsafe fn reported_error(
    operation: &'static str,
    out_error: *mut c_char,
    reported: impl FnOnce(String) -> ChatError,
) -> ChatError {
    unsafe { CppAllocatedString::from_raw(out_error) }
        .into_string()
        .map_or_else(
            |source| ChatError::ExceptionMessageNotUtf8 { operation, source },
            reported,
        )
}

#[cfg(test)]
mod tests {
    use super::reported_error;
    use crate::chat_error::ChatError;

    #[test]
    fn an_exception_message_that_is_not_utf8_is_reported_with_its_bytes() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"\xff".as_ptr()) };

        assert_eq!(
            unsafe {
                reported_error("llama_rs_chat_session_read", out_error, |message| {
                    ChatError::ParseReported { message }
                })
            },
            ChatError::ExceptionMessageNotUtf8 {
                operation: "llama_rs_chat_session_read",
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            }
        );
    }
}
