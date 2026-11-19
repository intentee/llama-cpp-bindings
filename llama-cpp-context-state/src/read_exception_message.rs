use std::ffi::c_char;

use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::context_state_error::ContextStateError;

/// # Safety
///
/// `out_error` must be the non-null exception message the C++ side allocated for a status that
/// carries one; it is freed here.
pub unsafe fn read_exception_message(
    operation: &'static str,
    out_error: *mut c_char,
) -> Result<String, ContextStateError> {
    unsafe { CppAllocatedString::from_raw(out_error) }
        .into_string()
        .map_err(|source| ContextStateError::ExceptionMessageNotUtf8 { operation, source })
}

#[cfg(test)]
mod tests {
    use super::read_exception_message;
    use crate::context_state_error::ContextStateError;

    #[test]
    fn exception_message_that_is_not_utf8_keeps_its_bytes() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"\xff".as_ptr()) };

        assert_eq!(
            unsafe { read_exception_message("llama_rs_state_get_data", out_error) },
            Err(ContextStateError::ExceptionMessageNotUtf8 {
                operation: "llama_rs_state_get_data",
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            })
        );
    }
}
