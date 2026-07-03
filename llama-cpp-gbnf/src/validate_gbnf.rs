use std::ffi::CString;

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
        LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION => {
            Err(GbnfValidationError::GrammarEngineThrew)
        }
        other => unreachable!("llama_rs_validate_gbnf returned unrecognized status {other}"),
    }
}

/// # Errors
///
/// Returns [`GbnfValidationError`] when `grammar` or `root` contains an interior
/// NUL byte, or when the grammar parser rejects the grammar.
pub fn validate_gbnf(grammar: &str, root: &str) -> Result<(), GbnfValidationError> {
    let grammar_cstring = CString::new(grammar).map_err(GbnfValidationError::GrammarContainsNul)?;
    let root_cstring = CString::new(root).map_err(GbnfValidationError::RootContainsNul)?;

    let status = unsafe { llama_rs_validate_gbnf(grammar_cstring.as_ptr(), root_cstring.as_ptr()) };

    validation_status_to_result(status, root)
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;

    use llama_cpp_bindings_sys::LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
    use llama_cpp_bindings_sys::llama_rs_gbnf_validation_status;

    use super::validate_gbnf;
    use super::validation_status_to_result;
    use crate::gbnf_validation_error::GbnfValidationError;

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
    fn exception_status_maps_to_grammar_engine_threw() {
        assert_eq!(
            validation_status_to_result(LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION, "root"),
            Err(GbnfValidationError::GrammarEngineThrew)
        );
    }

    #[test]
    #[should_panic(expected = "unrecognized status")]
    fn unrecognized_status_panics() {
        let _ = validation_status_to_result(llama_rs_gbnf_validation_status::MAX, "root");
    }
}
