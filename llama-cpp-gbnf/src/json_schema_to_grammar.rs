use std::ffi::{CString, c_char};
use std::ptr;

use llama_cpp_bindings_sys::bindings::llama_rs_json_schema_to_grammar_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::gbnf_error::GbnfError;
use crate::read_exception_message::read_exception_message;

const OPERATION: &str = "llama_rs_json_schema_to_grammar";

/// # Safety
///
/// On `LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK` `out` must be the non-null grammar the C++ side
/// allocated, and on the statuses that carry a message `error_ptr` must be the non-null message
/// it allocated; whichever is read is freed here.
unsafe fn json_schema_to_grammar_status_to_result(
    status: llama_rs_json_schema_to_grammar_status,
    out: *mut c_char,
    error_ptr: *mut c_char,
) -> Result<String, GbnfError> {
    match status {
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK => {
            unsafe { CppAllocatedString::from_raw(out) }
                .into_string()
                .map_err(|source| GbnfError::GrammarNotUtf8 { source })
        }
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GbnfError::SchemaConversionNotEnoughMemory)
        }
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GbnfError::SchemaConversionLlamaCppOutOfMemory)
        }
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_SCHEMA => {
            let message = unsafe { read_exception_message(OPERATION, error_ptr) }?;
            Err(GbnfError::InvalidSchema { message })
        }
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_INVALID_JSON => {
            let message = unsafe { read_exception_message(OPERATION, error_ptr) }?;
            Err(GbnfError::InvalidJson { message })
        }
        llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_exception_message(OPERATION, error_ptr) }?;
            Err(GbnfError::SchemaConversionReported { message })
        }
    }
}

/// # Errors
///
/// Returns [`GbnfError`] if the schema string contains a NUL byte,
/// llama.cpp rejects the schema or fails to convert it, or the returned grammar or error
/// message is not valid UTF-8.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String, GbnfError> {
    let schema_cstr = CString::new(schema_json).map_err(GbnfError::SchemaContainsNulByte)?;
    let mut out: *mut c_char = ptr::null_mut();
    let mut error_ptr: *mut c_char = ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::bindings::llama_rs_json_schema_to_grammar(
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

    use llama_cpp_bindings_sys::bindings::llama_rs_json_schema_to_grammar_status;

    use super::json_schema_to_grammar_status_to_result;
    use crate::gbnf_error::GbnfError;
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
    fn llama_cpp_exception_status_carries_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"conversion threw".as_ptr())
        };

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
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
                llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_ERROR_STRING_ALLOCATION_FAILED,
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
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                invalid_utf8_grammar.as_ptr().cast::<c_char>(),
            )
        };
        assert!(!out.is_null(), "llama_rs_string_dup must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                ptr::null_mut(),
            )
        };
        assert_eq!(
            result,
            Err(GbnfError::GrammarNotUtf8 {
                source: String::from_utf8(vec![0xFF]).unwrap_err(),
            })
        );
    }

    #[test]
    fn ok_status_with_valid_utf8_grammar_returns_grammar_string() {
        let grammar_text: &[u8; 14] = b"root ::= \"x\"\0\0";
        let out = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                grammar_text.as_ptr().cast::<c_char>(),
            )
        };
        assert!(!out.is_null(), "llama_rs_string_dup must allocate a copy");

        let result = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_OK,
                out,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok("root ::= \"x\"".to_owned()));
    }

    #[test]
    fn json_schema_to_grammar_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            json_schema_to_grammar_status_to_result(
                llama_rs_json_schema_to_grammar_status::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_OUT_OF_MEMORY,
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
