use std::ffi::CStr;
use std::ffi::c_char;

use crate::ffi_error::FfiError;

/// One function of the C++ wrapper, named so that every failure of its FFI contract says which
/// call broke it.
#[derive(Clone, Copy, Debug)]
pub struct WrapperOperation {
    name: &'static str,
}

impl WrapperOperation {
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// The failure of a wrapper that answered outside its contract: `detail` says how.
    #[must_use]
    pub const fn contract_violated(self, detail: &'static str) -> FfiError {
        FfiError::ContractViolated {
            operation: self.name,
            detail,
        }
    }

    /// Takes ownership of a C string the wrapper stored in one of its out-parameters and
    /// hands the allocation back to the wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`FfiError::ContractViolated`] when `string_ptr` is null. A wrapper always fills the
    /// slot it says it filled, so a null pointer means the wrapper broke its own contract rather
    /// than that the string is unknown. Returns [`FfiError::TextNotUtf8`], keeping the bytes,
    /// when the string is not valid UTF-8.
    ///
    /// # Safety
    ///
    /// `string_ptr` must be either null or a valid pointer to a null-terminated C string
    /// allocated by `llama_rs_dup_string`.
    pub unsafe fn read_and_free_cpp_string(
        self,
        string_ptr: *mut c_char,
        detail_when_missing: &'static str,
    ) -> Result<String, FfiError> {
        if string_ptr.is_null() {
            return Err(self.contract_violated(detail_when_missing));
        }

        let bytes = unsafe { CStr::from_ptr(string_ptr) }.to_bytes().to_vec();

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(string_ptr) };

        String::from_utf8(bytes).map_err(|not_utf8| FfiError::TextNotUtf8 {
            operation: self.name,
            source: not_utf8.utf8_error(),
            bytes: not_utf8.into_bytes(),
        })
    }

    /// The failure of a wrapper that answered with a status `code` it does not define.
    #[must_use]
    pub const fn unknown_status(self, code: i64) -> FfiError {
        FfiError::UnknownStatus {
            operation: self.name,
            code,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::WrapperOperation;
    use crate::ffi_error::FfiError;

    const JSON_SCHEMA_TO_GRAMMAR: WrapperOperation =
        WrapperOperation::new("llama_rs_json_schema_to_grammar");

    #[test]
    fn a_broken_contract_names_the_operation_and_how_it_was_broken() {
        assert_eq!(
            JSON_SCHEMA_TO_GRAMMAR.contract_violated("was given a null schema_json argument"),
            FfiError::ContractViolated {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "was given a null schema_json argument",
            }
        );
    }

    #[test]
    fn a_status_the_wrapper_does_not_define_keeps_its_code() {
        assert_eq!(
            JSON_SCHEMA_TO_GRAMMAR.unknown_status(255),
            FfiError::UnknownStatus {
                operation: "llama_rs_json_schema_to_grammar",
                code: 255,
            }
        );
    }

    #[test]
    fn reads_and_reclaims_a_string_allocated_by_the_wrapper() {
        let message = unsafe {
            JSON_SCHEMA_TO_GRAMMAR.read_and_free_cpp_string(
                wrapper_allocated_error_message_pointer(),
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
            JSON_SCHEMA_TO_GRAMMAR.read_and_free_cpp_string(
                ptr::null_mut(),
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            })
        );
    }

    #[test]
    fn a_string_that_is_not_utf8_keeps_its_bytes() {
        let not_utf8 = c"llama\xff";
        let result = unsafe {
            JSON_SCHEMA_TO_GRAMMAR.read_and_free_cpp_string(
                llama_cpp_bindings_sys::llama_rs_string_dup(not_utf8.as_ptr()),
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            result,
            Err(FfiError::TextNotUtf8 {
                operation: "llama_rs_json_schema_to_grammar",
                bytes: not_utf8.to_bytes().to_vec(),
                source: not_utf8.to_str().expect_err("0xff is not UTF-8"),
            })
        );
    }
}
