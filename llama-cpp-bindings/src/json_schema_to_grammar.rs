use std::ffi::{CStr, CString, c_char};
use std::ptr;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;

/// # Safety
///
/// On `LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK` the function reads and frees `out` as a
/// null-terminated C string allocated by the wrapper, so `out` must be a valid such
/// pointer for that status. On error statuses it reads and frees `error_ptr` via
/// [`read_and_free_cpp_error`], which tolerates a null pointer.
unsafe fn json_schema_to_grammar_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar_status,
    out: *mut c_char,
    error_ptr: *mut c_char,
) -> Result<String, JsonSchemaToGrammarError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK => {
            if out.is_null() {
                return Err(FfiContractError {
                    operation: "llama_rs_json_schema_to_grammar",
                    detail: "success status contained a null grammar string",
                }
                .into());
            }
            let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };
            Ok(String::from_utf8(grammar_bytes)?)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(JsonSchemaToGrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(JsonSchemaToGrammarError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_json_schema_to_grammar",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(JsonSchemaToGrammarError::InvalidSchema { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_JSON => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_json_schema_to_grammar",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(JsonSchemaToGrammarError::InvalidJson { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    error_ptr,
                    "llama_rs_json_schema_to_grammar",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(JsonSchemaToGrammarError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "was given a null schema_json argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "was given a null out_grammar argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_json_schema_to_grammar",
            code: i64::from(other),
        }
        .into()),
    }
}

/// # Errors
///
/// Returns [`JsonSchemaToGrammarError`] if the schema string contains a NUL byte,
/// the wrapper reports any non-OK status, or the returned grammar is not valid UTF-8.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String, JsonSchemaToGrammarError> {
    let schema_cstr = CString::new(schema_json)?;
    let mut out: *mut c_char = ptr::null_mut();
    let mut error_ptr: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema_cstr.as_ptr(),
            &raw mut out,
            &raw mut error_ptr,
        )
    };

    unsafe { json_schema_to_grammar_status_to_result(status, out, error_ptr) }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;
    use crate::json_schema_to_grammar::json_schema_to_grammar;

    #[test]
    fn a_schema_with_a_nul_byte_is_reported_before_conversion() {
        let schema = "{\x00}";

        assert_eq!(
            json_schema_to_grammar(schema),
            Err(JsonSchemaToGrammarError::SchemaContainsNulByte(
                CString::new(schema).unwrap_err()
            ))
        );
    }

    #[test]
    fn a_string_schema_converts_to_its_gbnf_grammar() {
        let schema = r#"{"type": "string"}"#;

        assert_eq!(
            json_schema_to_grammar(schema),
            Ok(concat!(
                r#"char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#,
                "\n",
                r#"root ::= "\"" char* "\"""#,
                "\n",
                r#"space ::= | " " | "\n"{1,2} [ \t]{0,20}"#,
                "\n",
            )
            .to_owned())
        );
    }

    #[test]
    fn invalid_json_is_reported_with_the_parser_message() {
        let schema = "not valid json at all";

        assert_eq!(
            json_schema_to_grammar(schema),
            Err(JsonSchemaToGrammarError::InvalidJson {
                message: "[json.exception.parse_error.101] parse error at line 1, column 2: \
                          syntax error while parsing value - invalid literal; last read: 'no'"
                    .to_owned(),
            })
        );
    }

    #[test]
    fn an_unresolved_ref_is_reported_as_an_invalid_schema_with_its_message() {
        let schema = r##"{"$ref": "#/$defs/Missing"}"##;

        assert_eq!(
            json_schema_to_grammar(schema),
            Err(JsonSchemaToGrammarError::InvalidSchema {
                message: "JSON schema conversion failed:\nJSON schema error at #: \
                          cannot resolve $ref #/$defs/Missing, $defs not found"
                    .to_owned(),
            })
        );
    }
}

#[cfg(test)]
mod json_schema_to_grammar_status_to_result_tests {
    use llama_cpp_ffi_status::cpp_string_error::CppStringError;
    use std::ffi::c_char;
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::json_schema_to_grammar_status_to_result;
    use crate::error::json_schema_to_grammar_error::JsonSchemaToGrammarError;

    #[test]
    fn invalid_schema_status_without_a_message_is_a_contract_error() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            })
            .into())
        );
    }

    #[test]
    fn invalid_json_status_without_a_message_is_a_contract_error() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_JSON,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            })
            .into())
        );
    }

    #[test]
    fn llama_cpp_exception_status_without_a_message_is_a_contract_error() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(CppStringError::Missing(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            })
            .into())
        );
    }

    #[test]
    fn llama_cpp_exception_status_carries_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"conversion threw".as_ptr()) };

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::Reported {
                message: "conversion threw".to_owned(),
            })
        );
    }

    #[test]
    fn allocation_failed_status_returns_not_enough_memory() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(JsonSchemaToGrammarError::NotEnoughMemory));
    }

    #[test]
    fn ok_status_with_non_utf8_grammar_reports_the_utf8_error() {
        let invalid_utf8_grammar: [u8; 2] = [0xFF, 0];
        let out = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(
                invalid_utf8_grammar.as_ptr().cast::<c_char>(),
            )
        };
        assert!(!out.is_null(), "the wrapper must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::GrammarNotUtf8(
                String::from_utf8(vec![0xFF]).unwrap_err()
            ))
        );
    }

    #[test]
    fn ok_status_with_valid_utf8_grammar_returns_grammar_string() {
        let grammar_text: &[u8; 14] = b"root ::= \"x\"\0\0";
        let out = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(grammar_text.as_ptr().cast::<c_char>())
        };
        assert!(!out.is_null(), "the wrapper must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok("root ::= \"x\"".to_owned()));
    }

    #[test]
    fn ok_status_with_null_grammar_is_contract_error() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::FfiContract(FfiContractError {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "success status contained a null grammar string",
            }))
        );
    }

    #[test]
    fn unknown_status_is_preserved() {
        let result = unsafe {
            json_schema_to_grammar_status_to_result(255, ptr::null_mut(), ptr::null_mut())
        };

        assert_eq!(
            result,
            Err(JsonSchemaToGrammarError::FfiStatus(FfiStatusError {
                operation: "llama_rs_json_schema_to_grammar",
                code: 255,
            }))
        );
    }

    #[test]
    fn json_schema_to_grammar_status_to_result_reports_a_null_schema_json_arg_as_a_contract_violation()
     {
        let outcome = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_json_schema_to_grammar",
                    detail: "was given a null schema_json argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn json_schema_to_grammar_status_to_result_reports_a_null_out_grammar_arg_as_a_contract_violation()
     {
        let outcome = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_json_schema_to_grammar",
                    detail: "was given a null out_grammar argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn json_schema_to_grammar_status_to_result_reports_a_null_out_error_arg_as_a_contract_violation()
     {
        let outcome = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_json_schema_to_grammar",
                    detail: "was given a null out_error argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn json_schema_to_grammar_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(
            outcome.err(),
            Some(JsonSchemaToGrammarError::LlamaCppOutOfMemory)
        );
    }
}
