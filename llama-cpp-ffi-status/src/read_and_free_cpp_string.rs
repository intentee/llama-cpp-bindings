use std::ffi::CStr;
use std::ffi::c_char;

use crate::cpp_string_error::CppStringError;
use crate::ffi_contract_error::FfiContractError;

fn utf8_text_of(bytes: Vec<u8>, operation: &'static str) -> Result<String, CppStringError> {
    String::from_utf8(bytes).map_err(|not_utf8| CppStringError::NotUtf8 {
        operation,
        source: not_utf8.utf8_error(),
        bytes: not_utf8.into_bytes(),
    })
}

/// Takes ownership of a C string a wrapper stored in one of its out-parameters and
/// hands the allocation back to the wrapper.
///
/// # Errors
///
/// Returns [`CppStringError::Missing`] when `string_ptr` is null. A wrapper always fills the
/// slot it says it filled, so a null pointer means the wrapper broke its own contract rather
/// than that the string is unknown. Returns [`CppStringError::NotUtf8`], keeping the bytes,
/// when the string is not valid UTF-8.
///
/// # Safety
///
/// `string_ptr` must be either null or a valid pointer to a null-terminated C string
/// allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_cpp_string(
    string_ptr: *mut c_char,
    operation: &'static str,
    detail_when_missing: &'static str,
) -> Result<String, CppStringError> {
    if string_ptr.is_null() {
        return Err(FfiContractError {
            operation,
            detail: detail_when_missing,
        }
        .into());
    }

    let bytes = unsafe { CStr::from_ptr(string_ptr) }.to_bytes().to_vec();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(string_ptr) };

    utf8_text_of(bytes, operation)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::read_and_free_cpp_string;
    use crate::cpp_string_error::CppStringError;
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
            message,
            Ok(
                "[json.exception.parse_error.101] parse error at line 1, column 2: syntax error \
                while parsing value - invalid literal; last read: 'no'"
                    .to_owned()
            ),
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
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            }))
        );
    }

    #[test]
    fn a_string_that_is_not_utf8_keeps_its_bytes() {
        let not_utf8 = c"llama\xff";
        let result = unsafe {
            read_and_free_cpp_string(
                llama_cpp_bindings_sys::llama_rs_string_dup(not_utf8.as_ptr()),
                "llama_rs_json_schema_to_grammar",
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            result,
            Err(CppStringError::NotUtf8 {
                operation: "llama_rs_json_schema_to_grammar",
                bytes: not_utf8.to_bytes().to_vec(),
                source: not_utf8.to_str().expect_err("0xff is not UTF-8"),
            })
        );
    }
}
