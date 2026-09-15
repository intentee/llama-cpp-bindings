use std::ffi::CStr;
use std::ffi::c_char;

use crate::ffi_contract_error::FfiContractError;

/// Takes ownership of a C string a wrapper stored in one of its out-parameters and
/// hands the allocation back to the wrapper.
///
/// # Errors
///
/// Returns [`FfiContractError`] when `string_ptr` is null. A wrapper always fills the
/// slot it says it filled, so a null pointer means the wrapper broke its own contract
/// rather than that the string is unknown.
///
/// # Safety
///
/// `string_ptr` must be either null or a valid pointer to a null-terminated C string
/// allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_cpp_string(
    string_ptr: *mut c_char,
    operation: &'static str,
    detail_when_missing: &'static str,
) -> Result<String, FfiContractError> {
    if string_ptr.is_null() {
        return Err(FfiContractError {
            operation,
            detail: detail_when_missing,
        });
    }

    let value = unsafe { CStr::from_ptr(string_ptr) }
        .to_string_lossy()
        .into_owned();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(string_ptr) };

    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer;

    use super::read_and_free_cpp_string;
    use crate::ffi_contract_error::FfiContractError;

    #[test]
    fn reads_and_reclaims_a_string_allocated_by_the_wrapper() {
        let message = unsafe {
            read_and_free_cpp_string(
                wrapper_allocated_error_message_pointer(),
                "llama_rs_json_schema_to_grammar",
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            message.map(|text| text.contains("parse error")),
            Ok(true),
            "the llama.cpp json parser reports its failure through the error slot"
        );
    }

    #[test]
    fn a_missing_string_is_a_contract_violation() {
        let result = unsafe {
            read_and_free_cpp_string(
                ptr::null_mut(),
                "llama_rs_json_schema_to_grammar",
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            result,
            Err(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            })
        );
    }
}
