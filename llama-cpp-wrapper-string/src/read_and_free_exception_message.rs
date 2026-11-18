use std::ffi::c_char;

use crate::read_and_free_wrapper_string::read_and_free_wrapper_string;
use crate::wrapper_string_error::WrapperStringError;

/// Takes ownership of the message of a thrown C++ exception that the wrapper `operation`
/// reported in its error out-parameter.
///
/// # Errors
///
/// Returns [`WrapperStringError::ExceptionMessageMissing`] when the wrapper reported the
/// exception without a message, or [`WrapperStringError::NotUtf8`] when the message is not valid
/// UTF-8.
///
/// # Safety
///
/// `message_ptr` must be either null or a valid pointer to a null-terminated C string
/// allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_exception_message(
    operation: &'static str,
    message_ptr: *mut c_char,
) -> Result<String, WrapperStringError> {
    unsafe { read_and_free_wrapper_string(operation, message_ptr) }.and_then(|message| {
        message.ok_or(WrapperStringError::ExceptionMessageMissing { operation })
    })
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::read_and_free_exception_message;
    use crate::wrapper_string_error::WrapperStringError;

    #[test]
    fn a_thrown_exception_reported_without_a_message_is_a_missing_message() {
        let result = unsafe {
            read_and_free_exception_message("llama_rs_json_schema_to_grammar", ptr::null_mut())
        };

        assert_eq!(
            result,
            Err(WrapperStringError::ExceptionMessageMissing {
                operation: "llama_rs_json_schema_to_grammar",
            })
        );
    }
}
