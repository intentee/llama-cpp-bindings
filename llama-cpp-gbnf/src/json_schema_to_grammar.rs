use std::ffi::{CStr, CString, c_char};
use std::ptr;

use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;

use crate::gbnf_error::GbnfError;

/// # Safety
///
/// On `LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK` the function reads and frees `out` as a
/// null-terminated C string allocated by the wrapper, so `out` must be a valid such
/// pointer for that status. On error statuses it reads and frees `error_ptr`, reporting a null
/// pointer as a contract violation.
unsafe fn json_schema_to_grammar_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar_status,
    out: *mut c_char,
    error_ptr: *mut c_char,
) -> Result<String, GbnfError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_json_schema_to_grammar");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK => {
            if out.is_null() {
                return Err(OPERATION
                    .contract_violated("success status contained a null grammar string")
                    .into());
            }
            let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };
            String::from_utf8(grammar_bytes).map_err(GbnfError::GrammarNotUtf8)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GbnfError::SchemaConversionNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GbnfError::SchemaConversionLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA => {
            let message = unsafe { OPERATION.read_and_free_exception_message(error_ptr) }?;
            Err(GbnfError::InvalidSchema { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_JSON => {
            let message = unsafe { OPERATION.read_and_free_exception_message(error_ptr) }?;
            Err(GbnfError::InvalidJson { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(error_ptr) }?;
            Err(GbnfError::SchemaConversionReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_SCHEMA_JSON_ARG => {
            Err(OPERATION
                .contract_violated("was given a null schema_json argument")
                .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_GRAMMAR_ARG => {
            Err(OPERATION
                .contract_violated("was given a null out_grammar argument")
                .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_NULL_OUT_ERROR_ARG => {
            Err(OPERATION
                .contract_violated("was given a null out_error argument")
                .into())
        }
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

/// # Errors
///
/// Returns [`GbnfError`] if the schema string contains a NUL byte,
/// the wrapper reports any non-OK status, or the returned grammar is not valid UTF-8.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String, GbnfError> {
    let schema_cstr = CString::new(schema_json).map_err(GbnfError::SchemaContainsNulByte)?;
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
    use std::ffi::c_char;

    use std::ptr;

    use llama_cpp_ffi_status::ffi_error::FfiError;

    use crate::gbnf_error::GbnfError;

    use super::json_schema_to_grammar_status_to_result;
    use crate::json_schema_to_grammar::json_schema_to_grammar;

    #[test]
    fn a_schema_with_a_nul_byte_is_reported_before_conversion() {
        let schema = "{\x00}";

        assert_eq!(
            json_schema_to_grammar(schema),
            Err(GbnfError::SchemaContainsNulByte(
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
            Err(GbnfError::InvalidJson {
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
            Err(GbnfError::InvalidSchema {
                message: "JSON schema conversion failed:\nJSON schema error at #: \
                          cannot resolve $ref #/$defs/Missing, $defs not found"
                    .to_owned(),
            })
        );
    }

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
            Err(FfiError::ContractViolated {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            }
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
            Err(FfiError::ContractViolated {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            }
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
            Err(FfiError::ContractViolated {
                operation: "llama_rs_json_schema_to_grammar",
                detail: "reported a thrown C++ exception without an error message",
            }
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
            Err(GbnfError::SchemaConversionReported {
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

        assert_eq!(result, Err(GbnfError::SchemaConversionNotEnoughMemory));
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
            Err(GbnfError::GrammarNotUtf8(
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
            Err(GbnfError::Ffi(FfiError::ContractViolated {
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
            Err(GbnfError::Ffi(FfiError::UnknownStatus {
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
                FfiError::ContractViolated {
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
                FfiError::ContractViolated {
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
                FfiError::ContractViolated {
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
            Some(GbnfError::SchemaConversionLlamaCppOutOfMemory)
        );
    }
}
