use std::ffi::CString;
use std::ptr;

use llama_cpp_gbnf_sys::LlamaRsGbnfParseStatus;
use llama_cpp_gbnf_sys::llama_grammar;
use llama_cpp_gbnf_sys::llama_rs_gbnf_free;
use llama_cpp_gbnf_sys::llama_rs_gbnf_parse;

use crate::gbnf_error::GbnfError;
use crate::gbnf_matcher::GbnfMatcher;
use crate::validation::Validation;

fn parse_status_to_result(
    status: LlamaRsGbnfParseStatus,
    grammar: *mut llama_grammar,
    root: &str,
) -> Result<*mut llama_grammar, GbnfError> {
    match status {
        LlamaRsGbnfParseStatus::Ok => Ok(grammar),
        LlamaRsGbnfParseStatus::SyntaxError => Err(GbnfError::SyntaxError),
        LlamaRsGbnfParseStatus::EmptyRuleSet => Err(GbnfError::EmptyRuleSet),
        LlamaRsGbnfParseStatus::RootSymbolMissing => Err(GbnfError::RootSymbolMissing {
            root: root.to_owned(),
        }),
        LlamaRsGbnfParseStatus::LeftRecursion => Err(GbnfError::LeftRecursion),
        LlamaRsGbnfParseStatus::AllocationFailed | LlamaRsGbnfParseStatus::ThrewCxxException => {
            Err(GbnfError::AllocationFailed)
        }
    }
}

#[derive(Debug)]
pub struct GbnfGrammar {
    handle: *mut llama_grammar,
}

impl GbnfGrammar {
    /// # Errors
    ///
    /// Returns [`GbnfError`] when the grammar or root contains a NUL byte, the
    /// grammar parser rejects the grammar, or the grammar engine cannot allocate.
    pub fn parse(grammar: &str, root: &str) -> Result<Self, GbnfError> {
        let grammar_cstring = CString::new(grammar).map_err(GbnfError::GrammarStringNulByte)?;
        let root_cstring = CString::new(root).map_err(GbnfError::RootNameNulByte)?;

        let mut handle: *mut llama_grammar = ptr::null_mut();

        let status = unsafe {
            llama_rs_gbnf_parse(
                grammar_cstring.as_ptr(),
                root_cstring.as_ptr(),
                &raw mut handle,
            )
        };

        Ok(Self {
            handle: parse_status_to_result(status, handle, root)?,
        })
    }

    /// # Errors
    ///
    /// Returns [`GbnfError::AllocationFailed`] when the grammar engine cannot
    /// allocate the matcher state.
    pub fn matcher(&self) -> Result<GbnfMatcher<'_>, GbnfError> {
        unsafe { GbnfMatcher::new(self.handle) }
    }

    /// # Errors
    ///
    /// Returns [`GbnfError::AllocationFailed`] when the grammar engine cannot
    /// allocate while matching `text`.
    pub fn matches(&self, text: &str) -> Result<Validation, GbnfError> {
        let mut matcher = self.matcher()?;

        matcher.feed_str(text)?;

        if !matcher.is_rejected() && matcher.is_accepting() {
            Ok(Validation::Accepted)
        } else {
            Ok(Validation::Rejected)
        }
    }
}

impl Drop for GbnfGrammar {
    fn drop(&mut self) {
        unsafe { llama_rs_gbnf_free(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::mem::discriminant;

    use super::GbnfGrammar;
    use crate::gbnf_error::GbnfError;
    use crate::validation::Validation;

    #[test]
    fn empty_grammar_returns_empty_rule_set() {
        let error = GbnfGrammar::parse("", "root").expect_err("empty grammar is rejected");

        assert_eq!(error, GbnfError::EmptyRuleSet);
    }

    #[test]
    fn left_recursive_grammar_returns_left_recursion() {
        let error = GbnfGrammar::parse(r#"root ::= root "x""#, "root")
            .expect_err("left recursion is rejected");

        assert_eq!(error, GbnfError::LeftRecursion);
    }

    #[test]
    fn valid_grammar_parses() {
        GbnfGrammar::parse(r#"root ::= "yes" | "no""#, "root").expect("valid grammar parses");
    }

    #[test]
    fn malformed_grammar_returns_syntax_error() {
        let error = GbnfGrammar::parse("root ::= (", "root").expect_err("malformed is rejected");

        assert_eq!(error, GbnfError::SyntaxError);
    }

    #[test]
    fn missing_root_returns_root_symbol_missing() {
        let error =
            GbnfGrammar::parse(r#"expr ::= "x""#, "root").expect_err("missing root is rejected");

        assert_eq!(
            error,
            GbnfError::RootSymbolMissing {
                root: "root".to_owned()
            }
        );
    }

    #[test]
    fn grammar_with_nul_byte_returns_grammar_string_nul_byte() {
        let error =
            GbnfGrammar::parse("root ::= \"a\0b\"", "root").expect_err("nul byte is rejected");
        let representative =
            GbnfError::GrammarStringNulByte(CString::new(b"a\0b".to_vec()).unwrap_err());

        assert_eq!(discriminant(&error), discriminant(&representative));
    }

    #[test]
    fn root_with_nul_byte_returns_root_name_nul_byte() {
        let error =
            GbnfGrammar::parse(r#"root ::= "x""#, "ro\0ot").expect_err("nul byte is rejected");
        let representative =
            GbnfError::RootNameNulByte(CString::new(b"ro\0ot".to_vec()).unwrap_err());

        assert_eq!(discriminant(&error), discriminant(&representative));
    }

    #[test]
    fn matches_accepts_text_in_the_language() {
        let grammar = GbnfGrammar::parse(r"root ::= [0-9]+", "root").expect("grammar parses");

        assert_eq!(
            grammar.matches("123").expect("engine allocates"),
            Validation::Accepted
        );
    }

    #[test]
    fn matches_rejects_text_outside_the_language() {
        let grammar = GbnfGrammar::parse(r"root ::= [0-9]+", "root").expect("grammar parses");

        assert_eq!(
            grammar.matches("12a").expect("engine allocates"),
            Validation::Rejected
        );
    }

    #[test]
    fn matches_rejects_incomplete_text() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");

        assert_eq!(
            grammar.matches("ye").expect("engine allocates"),
            Validation::Rejected
        );
    }

    #[cfg(feature = "fault-injection")]
    #[test]
    fn parse_reports_allocation_failure_when_building_the_grammar() {
        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_arm(0) };
        let result = GbnfGrammar::parse(r#"root ::= "ok""#, "root");
        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_disarm() };

        assert_eq!(
            result.expect_err("building the grammar fails under an armed fault"),
            GbnfError::AllocationFailed
        );
    }

    #[cfg(feature = "fault-injection")]
    #[test]
    fn matches_reports_allocation_failure_when_cloning_the_matcher() {
        let grammar = GbnfGrammar::parse(r#"root ::= "ok""#, "root").expect("grammar parses");

        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_arm(0) };
        let result = grammar.matches("ok");
        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_disarm() };

        assert_eq!(
            result.expect_err("cloning the matcher fails under an armed fault"),
            GbnfError::AllocationFailed
        );
    }

    #[cfg(feature = "fault-injection")]
    #[test]
    fn matches_reports_allocation_failure_when_feeding_text() {
        let grammar = GbnfGrammar::parse(r#"root ::= "ok""#, "root").expect("grammar parses");

        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_arm(1) };
        let result = grammar.matches("ok");
        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_disarm() };

        assert_eq!(
            result.expect_err("feeding text fails under an armed fault"),
            GbnfError::AllocationFailed
        );
    }
}
