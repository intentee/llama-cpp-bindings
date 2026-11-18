use std::ffi::CStr;
use std::ffi::c_char;

use crate::wrapper_string_error::WrapperStringError;

/// Takes ownership of a C string that the wrapper `operation` stored in one of its
/// out-parameters and hands the allocation back to the wrapper; `None` when the slot is null.
///
/// # Errors
///
/// Returns [`WrapperStringError::NotUtf8`], keeping the bytes, when the string is not valid
/// UTF-8.
///
/// # Safety
///
/// `string_ptr` must be either null or a valid pointer to a null-terminated C string
/// allocated by `llama_rs_dup_string`.
pub unsafe fn read_and_free_wrapper_string(
    operation: &'static str,
    string_ptr: *mut c_char,
) -> Result<Option<String>, WrapperStringError> {
    if string_ptr.is_null() {
        return Ok(None);
    }

    let bytes = unsafe { CStr::from_ptr(string_ptr) }.to_bytes().to_vec();

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(string_ptr) };

    String::from_utf8(bytes)
        .map(Some)
        .map_err(|not_utf8| WrapperStringError::NotUtf8 {
            operation,
            source: not_utf8.utf8_error(),
            bytes: not_utf8.into_bytes(),
        })
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::read_and_free_wrapper_string;
    use crate::wrapper_string_error::WrapperStringError;

    const JSON_SCHEMA_TO_GRAMMAR: &str = "llama_rs_json_schema_to_grammar";

    #[test]
    fn reads_and_reclaims_a_string_allocated_by_the_wrapper() {
        let message = unsafe {
            read_and_free_wrapper_string(
                JSON_SCHEMA_TO_GRAMMAR,
                wrapper_allocated_error_message_pointer(),
            )
        };

        assert_eq!(
            message,
            Ok(Some(
                "[json.exception.parse_error.101] parse error at line 1, column 2: syntax error \
                while parsing value - invalid literal; last read: 'no'"
                    .to_owned()
            )),
            "the llama.cpp json parser reports its failure through the error slot"
        );
    }

    #[test]
    fn a_null_slot_holds_no_string() {
        assert_eq!(
            unsafe { read_and_free_wrapper_string(JSON_SCHEMA_TO_GRAMMAR, ptr::null_mut()) },
            Ok(None)
        );
    }

    #[test]
    fn a_string_that_is_not_utf8_keeps_its_bytes() {
        let not_utf8 = c"llama\xff";
        let result = unsafe {
            read_and_free_wrapper_string(
                JSON_SCHEMA_TO_GRAMMAR,
                llama_cpp_bindings_sys::llama_rs_string_dup(not_utf8.as_ptr()),
            )
        };

        assert_eq!(
            result,
            Err(WrapperStringError::NotUtf8 {
                operation: "llama_rs_json_schema_to_grammar",
                bytes: not_utf8.to_bytes().to_vec(),
                source: not_utf8.to_str().expect_err("0xff is not UTF-8"),
            })
        );
    }
}
