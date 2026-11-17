use std::ffi::{CString, c_char};
use std::ptr;

use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_OK;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_UNSUPPORTED_GRAMMAR;
use llama_cpp_bindings_sys::llama_rs_gbnf_validation_status;
use llama_cpp_bindings_sys::llama_rs_validate_gbnf;
use llama_cpp_bindings_sys::llama_vocab;
use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;

use crate::gbnf_error::GbnfError;

fn validation_status_to_result(
    status: llama_rs_gbnf_validation_status,
    root: &str,
    out_error: *mut c_char,
) -> Result<(), GbnfError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_validate_gbnf");

    match status {
        LLAMA_RS_GBNF_VALIDATION_OK => Ok(()),
        LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR => Err(GbnfError::SyntaxError),
        LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET => Err(GbnfError::EmptyRuleSet),
        LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING => Err(GbnfError::RootSymbolMissing {
            root: root.to_owned(),
        }),
        LLAMA_RS_GBNF_VALIDATION_UNSUPPORTED_GRAMMAR => Err(GbnfError::UnsupportedGrammar),
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_GRAMMAR_ARG => Err(OPERATION
            .contract_violated("grammar pointer was null")
            .into()),
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_ROOT_ARG => {
            Err(OPERATION.contract_violated("root pointer was null").into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_NULL_OUT_ERROR_ARG => Err(OPERATION
            .contract_violated("output error pointer was null")
            .into()),
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GbnfError::ValidationNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GbnfError::ValidationLlamaCppOutOfMemory)
        }
        LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                OPERATION.read_and_free_cpp_string(
                    out_error,
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(GbnfError::ValidationReported { message })
        }
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

/// # Errors
///
/// Returns [`GbnfError`] when `grammar` or `root` contains an interior
/// NUL byte, or when the grammar parser rejects the grammar.
pub fn validate_gbnf(
    vocab: Option<&llama_vocab>,
    grammar: &str,
    root: &str,
) -> Result<(), GbnfError> {
    let grammar_cstring = CString::new(grammar).map_err(GbnfError::GrammarContainsNul)?;
    let root_cstring = CString::new(root).map_err(GbnfError::RootContainsNul)?;

    let mut out_error = ptr::null_mut();
    let status = unsafe {
        llama_rs_validate_gbnf(
            vocab.map_or(ptr::null(), ptr::from_ref),
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
    use std::fmt::Write;
    use std::ptr;

    use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
    use llama_cpp_ffi_status::ffi_error::FfiError;

    use super::validate_gbnf;
    use super::validation_status_to_result;
    use crate::gbnf_error::GbnfError;

    #[test]
    fn valid_grammar_is_accepted() {
        assert_eq!(
            validate_gbnf(None, r#"root ::= "yes" | "no""#, "root"),
            Ok(())
        );
    }

    #[test]
    fn malformed_grammar_is_a_syntax_error() {
        assert_eq!(
            validate_gbnf(None, "root ::= (", "root"),
            Err(GbnfError::SyntaxError)
        );
    }

    #[test]
    fn empty_grammar_has_no_rules() {
        assert_eq!(
            validate_gbnf(None, "", "root"),
            Err(GbnfError::EmptyRuleSet)
        );
    }

    #[test]
    fn grammar_without_root_reports_missing_root() {
        assert_eq!(
            validate_gbnf(None, r#"expr ::= "x""#, "root"),
            Err(GbnfError::RootSymbolMissing {
                root: "root".to_owned()
            })
        );
    }

    #[test]
    fn left_recursive_grammar_is_rejected() {
        assert_eq!(
            validate_gbnf(None, r#"root ::= root "x""#, "root"),
            Err(GbnfError::UnsupportedGrammar)
        );
    }

    #[test]
    fn grammar_chaining_rules_past_the_recursion_limit_is_rejected() {
        let mut grammar = String::from("root ::= rule0\n");
        for rule_index in 0..300 {
            writeln!(grammar, "rule{rule_index} ::= rule{}", rule_index + 1).unwrap();
        }
        grammar.push_str("rule300 ::= \"x\"\n");

        assert_eq!(
            validate_gbnf(None, &grammar, "root"),
            Err(GbnfError::UnsupportedGrammar)
        );
    }

    #[test]
    fn grammar_with_interior_nul_is_reported() {
        let grammar = "root ::= \"a\0b\"";

        assert_eq!(
            validate_gbnf(None, grammar, "root").err(),
            CString::new(grammar)
                .err()
                .map(GbnfError::GrammarContainsNul)
        );
    }

    #[test]
    fn root_with_interior_nul_is_reported() {
        let root = "ro\0ot";

        assert_eq!(
            validate_gbnf(None, r#"root ::= "x""#, root).err(),
            CString::new(root).err().map(GbnfError::RootContainsNul)
        );
    }

    #[test]
    fn exception_status_without_message_is_a_contract_error() {
        assert_eq!(
            validation_status_to_result(
                LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
                "root",
                ptr::null_mut(),
            ),
            Err(GbnfError::Ffi(FfiError::ContractViolated {
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
            Err(GbnfError::ValidationReported {
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
                ptr::null_mut(),
            ),
            Err(GbnfError::Ffi(FfiError::ContractViolated {
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
                ptr::null_mut(),
            ),
            Err(GbnfError::Ffi(FfiError::ContractViolated {
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
                ptr::null_mut(),
            ),
            Err(GbnfError::Ffi(FfiError::ContractViolated {
                operation: "llama_rs_validate_gbnf",
                detail: "output error pointer was null",
            }))
        );
    }

    #[test]
    fn llama_cpp_out_of_memory_status_maps_to_llama_cpp_out_of_memory() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_LLAMA_CPP_OUT_OF_MEMORY,
                "root",
                ptr::null_mut(),
            ),
            Err(GbnfError::ValidationLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            validation_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED,
                "root",
                ptr::null_mut(),
            ),
            Err(GbnfError::ValidationNotEnoughMemory)
        );
    }

    #[test]
    fn unknown_status_is_preserved() {
        assert_eq!(
            validation_status_to_result(255, "root", ptr::null_mut(),),
            Err(GbnfError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_validate_gbnf",
                code: 255,
            }))
        );
    }
}
