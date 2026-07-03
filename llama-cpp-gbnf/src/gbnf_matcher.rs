use core::ffi::c_char;
use std::marker::PhantomData;
use std::ptr;

use llama_cpp_gbnf_sys::LlamaRsGbnfRuntimeStatus;
use llama_cpp_gbnf_sys::llama_grammar;
use llama_cpp_gbnf_sys::llama_rs_gbnf_accept_str;
use llama_cpp_gbnf_sys::llama_rs_gbnf_clone;
use llama_cpp_gbnf_sys::llama_rs_gbnf_free;
use llama_cpp_gbnf_sys::llama_rs_gbnf_is_accepting;
use llama_cpp_gbnf_sys::llama_rs_gbnf_is_rejected;

use crate::gbnf_error::GbnfError;

const fn runtime_status_to_result(status: LlamaRsGbnfRuntimeStatus) -> Result<(), GbnfError> {
    match status {
        LlamaRsGbnfRuntimeStatus::Ok => Ok(()),
        LlamaRsGbnfRuntimeStatus::AllocationFailed
        | LlamaRsGbnfRuntimeStatus::ThrewCxxException => Err(GbnfError::AllocationFailed),
    }
}

fn clone_handle(grammar: *const llama_grammar) -> Result<*mut llama_grammar, GbnfError> {
    let mut out_clone: *mut llama_grammar = ptr::null_mut();

    let status = unsafe { llama_rs_gbnf_clone(grammar, &raw mut out_clone) };

    runtime_status_to_result(status)?;

    Ok(out_clone)
}

pub struct GbnfMatcher<'grammar> {
    source: *mut llama_grammar,
    current: *mut llama_grammar,
    borrowed_grammar: PhantomData<&'grammar llama_grammar>,
}

impl GbnfMatcher<'_> {
    /// # Safety
    ///
    /// `source` must be a valid grammar handle that stays alive for the entire
    /// `'grammar` lifetime. The matcher clones it and never frees `source`.
    ///
    /// # Errors
    ///
    /// Returns [`GbnfError::AllocationFailed`] when the grammar engine cannot
    /// allocate the initial matcher state.
    pub(crate) unsafe fn new(source: *mut llama_grammar) -> Result<Self, GbnfError> {
        let current = clone_handle(source)?;

        Ok(Self {
            source,
            current,
            borrowed_grammar: PhantomData,
        })
    }

    /// # Errors
    ///
    /// Returns [`GbnfError::AllocationFailed`] when the grammar engine cannot
    /// allocate while advancing over `piece`.
    pub fn feed_str(&mut self, piece: &str) -> Result<(), GbnfError> {
        let status = unsafe {
            llama_rs_gbnf_accept_str(self.current, piece.as_ptr().cast::<c_char>(), piece.len())
        };

        runtime_status_to_result(status)
    }

    #[must_use]
    pub fn is_accepting(&self) -> bool {
        unsafe { llama_rs_gbnf_is_accepting(self.current) }
    }

    #[must_use]
    pub fn is_rejected(&self) -> bool {
        unsafe { llama_rs_gbnf_is_rejected(self.current) }
    }

    /// # Errors
    ///
    /// Returns [`GbnfError::AllocationFailed`] when the grammar engine cannot
    /// allocate a fresh copy of the initial matcher state.
    pub fn reset(&mut self) -> Result<(), GbnfError> {
        let fresh = clone_handle(self.source)?;

        unsafe { llama_rs_gbnf_free(self.current) };

        self.current = fresh;

        Ok(())
    }
}

impl Drop for GbnfMatcher<'_> {
    fn drop(&mut self) {
        unsafe { llama_rs_gbnf_free(self.current) };
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "fault-injection")]
    use crate::gbnf_error::GbnfError;
    use crate::gbnf_grammar::GbnfGrammar;

    #[test]
    fn incremental_feed_reaches_accepting_only_when_complete() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher().expect("matcher clones");

        matcher.feed_str("ye").expect("engine allocates");
        assert!(!matcher.is_accepting());
        assert!(!matcher.is_rejected());

        matcher.feed_str("s").expect("engine allocates");
        assert!(matcher.is_accepting());
    }

    #[test]
    fn feeding_a_violating_piece_rejects() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher().expect("matcher clones");

        matcher.feed_str("no").expect("engine allocates");
        assert!(matcher.is_rejected());
    }

    #[test]
    fn reset_restores_the_initial_state() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher().expect("matcher clones");

        matcher.feed_str("no").expect("engine allocates");
        assert!(matcher.is_rejected());

        matcher.reset().expect("engine allocates");
        assert!(!matcher.is_rejected());

        matcher.feed_str("yes").expect("engine allocates");
        assert!(matcher.is_accepting());
    }

    #[cfg(feature = "fault-injection")]
    #[test]
    fn reset_reports_allocation_failure_when_cloning_fails() {
        let grammar = GbnfGrammar::parse(r#"root ::= "yes""#, "root").expect("grammar parses");
        let mut matcher = grammar.matcher().expect("matcher clones");

        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_arm(0) };
        let result = matcher.reset();
        unsafe { llama_cpp_gbnf_sys::gbnf_alloc_fault_disarm() };

        assert_eq!(
            result.expect_err("cloning fails under an armed fault"),
            GbnfError::AllocationFailed
        );
    }
}
