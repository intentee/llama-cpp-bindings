use std::ffi::{CString, c_char};

use llama_cpp_ffi_status::FfiContractError;
use llama_cpp_ffi_status::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_LEFT_RECURSION;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_OK;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
use llama_cpp_bindings_sys::llama_rs_gbnf_validation_status;
use llama_cpp_bindings_sys::llama_rs_validate_gbnf;

use crate::gbnf_validation_error::GbnfValidationError;

fn validation_status_to_result(
    status: llama_rs_gbnf_validation_status,
    root: &str,
    out_error: *mut c_char,
) -> Result<(), GbnfValidationError> {
    match status {
        LLAMA_RS_GBNF_VALIDATION_OK => Ok(()),
        LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR => Err(GbnfValidationError::SyntaxError),
        LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET => Err(GbnfValidationError::EmptyRuleSet),
        LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING => {
            Err(GbnfValidationError::RootSymbolMissing {
                root: root.to_owned(),
            })
        }
        LLAMA_RS_GBNF_VALIDATION_LEFT_RECURSION => Err(GbnfValidationError::LeftRecursion),
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_GRAMMAR_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "grammar pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_ROOT_ARG => Err(FfiContractError {
            operation: "llama_rs_validate_gbnf",
            detail: "root pointer was null",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_OUT_ERROR_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "output error pointer was null",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GbnfValidationError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_VENDORED_OUT_OF_MEMORY => {
            Err(GbnfValidationError::VendoredOutOfMemory)
        }
        LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_validate_gbnf",
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(GbnfValidationError::Reported { message })
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_validate_gbnf",
            code: i64::from(other),
        }
        .into()),
    }
}

/// # Errors
///
/// Returns [`GbnfValidationError`] when `grammar` or `root` contains an interior
/// NUL byte, or when the grammar parser rejects the grammar.
pub fn validate_gbnf(grammar: &str, root: &str) -> Result<(), GbnfValidationError> {
    let grammar_cstring = CString::new(grammar).map_err(GbnfValidationError::GrammarContainsNul)?;
    let root_cstring = CString::new(root).map_err(GbnfValidationError::RootContainsNul)?;

    let mut out_error = std::ptr::null_mut();
    let status = unsafe {
        llama_rs_validate_gbnf(
            grammar_cstring.as_ptr(),
            root_cstring.as_ptr(),
            &raw mut out_error,
        )
    };

    validation_status_to_result(status, root, out_error)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use llama_cpp_ffi_status::FfiContractError;
    use llama_cpp_ffi_status::FfiStatusError;

    use super::validate_gbnf;
    use super::validation_status_to_result;
    use crate::gbnf_validation_error::GbnfValidationError;
    use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;

    #[test]
    fn valid_grammar_is_accepted() {
        assert_eq!(validate_gbnf(r#"root ::= "yes" | "no""#, "root"), Ok(()));
    }

    #[test]
    fn malformed_grammar_is_a_syntax_error() {
        assert_eq!(
            validate_gbnf("root ::= (", "root"),
            Err(GbnfValidationError::SyntaxError)
        );
    }

    #[test]
    fn empty_grammar_has_no_rules() {
        assert_eq!(
            validate_gbnf("", "root"),
            Err(GbnfValidationError::EmptyRuleSet)
        );
    }

    #[test]
    fn grammar_without_root_reports_missing_root() {
        assert_eq!(
            validate_gbnf(r#"expr ::= "x""#, "root"),
            Err(GbnfValidationError::RootSymbolMissing {
                root: "root".to_owned()
            })
        );
    }

    #[test]
    fn left_recursive_grammar_is_rejected() {
        assert_eq!(
            validate_gbnf(r#"root ::= root "x""#, "root"),
            Err(GbnfValidationError::LeftRecursion)
        );
    }

    #[test]
    fn grammar_with_interior_nul_is_reported() {
        let grammar = "root ::= \"a\0b\"";

        assert_eq!(
            validate_gbnf(grammar, "root").err(),
            CString::new(grammar)
                .err()
                .map(GbnfValidationError::GrammarContainsNul)
        );
    }

    #[test]
    fn root_with_interior_nul_is_reported() {
        let root = "ro\0ot";

        assert_eq!(
            validate_gbnf(r#"root ::= "x""#, root).err(),
            CString::new(root)
                .err()
                .map(GbnfValidationError::RootContainsNul)
        );
    }

    #[test]
    fn exception_status_without_message_maps_to_unknown_reported_error() {
        assert_eq!(
            validation_status_to_result(
                LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::FfiContract(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "reported a thrown C++ exception without an error message",
            }))
        );
    }

    #[test]
    fn exception_status_preserves_reported_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"grammar engine exploded".as_ptr())
        };
        assert!(!out_error.is_null());

        assert_eq!(
            validation_status_to_result(
                LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
                "root",
                out_error,
            ),
            Err(GbnfValidationError::Reported {
                message: "grammar engine exploded".to_owned()
            })
        );
    }

    #[test]
    fn null_grammar_status_is_contract_error() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_GRAMMAR_ARG,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::FfiContract(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "grammar pointer was null",
            }))
        );
    }

    #[test]
    fn null_root_status_is_contract_error() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_ROOT_ARG,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::FfiContract(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "root pointer was null",
            }))
        );
    }

    #[test]
    fn null_output_error_status_is_contract_error() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_OUT_ERROR_ARG,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::FfiContract(FfiContractError {
                operation: "llama_rs_validate_gbnf",
                detail: "output error pointer was null",
            }))
        );
    }

    #[test]
    fn vendored_out_of_memory_status_is_distinct_from_error_string_allocation() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_VENDORED_OUT_OF_MEMORY,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::VendoredOutOfMemory)
        );
    }

    #[test]
    fn allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED,
                "root",
                std::ptr::null_mut(),
            ),
            Err(GbnfValidationError::NotEnoughMemory)
        );
    }

    #[test]
    fn unknown_status_is_preserved() {
        assert_eq!(
            validation_status_to_result(255, "root", std::ptr::null_mut(),),
            Err(GbnfValidationError::FfiStatus(FfiStatusError {
                operation: "llama_rs_validate_gbnf",
                code: 255,
            }))
        );
    }
}
