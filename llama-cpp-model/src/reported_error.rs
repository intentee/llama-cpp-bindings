use std::ffi::c_char;

use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::model_error::ModelError;

/// Builds the error of a status that carries the message of a thrown C++ exception, or
/// [`ModelError::ExceptionMessageNotUtf8`] when that message is not valid UTF-8.
///
/// # Safety
///
/// `out_error` must be the non-null exception message the C++ side allocated for a status that
/// carries one; it is freed here.
pub unsafe fn reported_error(
    operation: &'static str,
    out_error: *mut c_char,
    reported: impl FnOnce(String) -> ModelError,
) -> ModelError {
    unsafe { CppAllocatedString::from_raw(out_error) }
        .into_string()
        .map_or_else(
            |source| ModelError::ExceptionMessageNotUtf8 { operation, source },
            reported,
        )
}

#[cfg(test)]
mod tests {
    use super::reported_error;
    use crate::model_error::ModelError;

    #[test]
    fn reported_error_carries_the_message_or_reports_that_it_is_not_utf8() {
        let reported = |message| ModelError::TokenizeReported { message };

        assert_eq!(
            [c"llama.cpp threw", c"\xff"].map(|message| unsafe {
                reported_error(
                    "llama_rs_tokenize",
                    llama_cpp_bindings_sys::bindings::llama_rs_string_dup(message.as_ptr()),
                    reported,
                )
            }),
            [
                ModelError::TokenizeReported {
                    message: "llama.cpp threw".to_owned(),
                },
                ModelError::ExceptionMessageNotUtf8 {
                    operation: "llama_rs_tokenize",
                    source: String::from_utf8(vec![0xFF]).unwrap_err(),
                },
            ]
        );
    }
}
