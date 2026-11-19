use std::ffi::{CString, c_char};
use std::ptr;

use llama_cpp_bindings_sys::bindings::llama_rs_gbnf_validation_status;
use llama_cpp_bindings_sys::bindings::llama_rs_validate_gbnf;
use llama_cpp_bindings_sys::bindings::llama_vocab;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::gbnf_error::GbnfError;

/// # Safety
///
/// On `LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION` `out_error` must be the non-null message the
/// C++ side allocated; it is freed here.
unsafe fn validation_status_to_result(
    status: llama_rs_gbnf_validation_status,
    root: &str,
    out_error: *mut c_char,
) -> Result<(), GbnfError> {
    match status {
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_OK => Ok(()),
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR => {
            Err(GbnfError::SyntaxError)
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET => {
            Err(GbnfError::EmptyRuleSet)
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING => {
            Err(GbnfError::RootSymbolMissing {
                root: root.to_owned(),
            })
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_UNSUPPORTED_GRAMMAR => {
            Err(GbnfError::UnsupportedGrammar)
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GbnfError::ValidationNotEnoughMemory)
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GbnfError::ValidationLlamaCppOutOfMemory)
        }
        llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_validate_gbnf", |message| {
                    GbnfError::ValidationReported { message }
                }))
        }
    }
}

/// # Errors
///
/// Returns [`GbnfError`] when `grammar` or `root` contains an interior
/// NUL byte, when the grammar parser rejects the grammar, or when llama.cpp throws while
/// validating it.
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

    unsafe { validation_status_to_result(status, root, out_error) }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::fmt::Write;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_gbnf_validation_status;

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
    fn exception_status_preserves_reported_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(
                c"grammar engine exploded".as_ptr(),
            )
        };
        assert!(!out_error.is_null());

        assert_eq!(
            unsafe {
                validation_status_to_result(
                    llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
                    "root",
                    out_error,
                )
            },
            Err(GbnfError::ValidationReported {
                message: "grammar engine exploded".to_owned()
            })
        );
    }

    #[test]
    fn llama_cpp_out_of_memory_status_maps_to_llama_cpp_out_of_memory() {
        assert_eq!(
            unsafe {
                validation_status_to_result(
                    llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_LLAMA_CPP_OUT_OF_MEMORY,
                    "root",
                    ptr::null_mut(),
                )
            },
            Err(GbnfError::ValidationLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn allocation_failed_status_maps_to_not_enough_memory() {
        assert_eq!(
            unsafe {
                validation_status_to_result(
                    llama_rs_gbnf_validation_status::LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED,
                    "root",
                    ptr::null_mut(),
                )
            },
            Err(GbnfError::ValidationNotEnoughMemory)
        );
    }
}
