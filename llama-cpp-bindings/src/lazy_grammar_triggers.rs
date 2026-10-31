use std::ffi::c_char;
use std::ptr;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::chat_grammar_trigger::ChatGrammarTrigger;
use crate::error::grammar_error::GrammarError;
use crate::token::LlamaToken;

fn anchored_pattern(pattern: &str) -> String {
    if pattern.is_empty() {
        return "^$".to_owned();
    }

    let start = if pattern.starts_with('^') { "" } else { "^" };
    let end = if pattern.ends_with('$') { "" } else { "$" };

    format!("{start}{pattern}{end}")
}

fn regex_escape_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_regex_escape_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, GrammarError> {
    const OPERATION: &str = "llama_rs_regex_escape";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_OK => Ok(unsafe {
            read_and_free_cpp_string(
                out_string,
                OPERATION,
                "success status contained a null string",
            )
        }?),
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(GrammarError::Reported {
                message: unsafe {
                    read_and_free_cpp_string(
                        out_error,
                        OPERATION,
                        "reported a thrown C++ exception without an error message",
                    )
                }?,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(GrammarError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(GrammarError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_TEXT_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: OPERATION,
                detail: "was given a null text argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_OUT_STRING_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: OPERATION,
                detail: "was given a null out_string argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_OUT_ERROR_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: OPERATION,
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(llama_cpp_ffi_status::FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

fn regex_escaped(word: &str) -> Result<String, GrammarError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_regex_escape(
            word.as_ptr().cast::<c_char>(),
            word.len(),
            &raw mut out_string,
            &raw mut out_error,
        )
    };

    regex_escape_status_to_result(status, out_string, out_error)
}

/// The trigger patterns and tokens that start a lazy grammar, built from a chat session's
/// grammar triggers the way llama.cpp's sampler builds them.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LazyGrammarTriggers {
    pub patterns: Vec<String>,
    pub tokens: Vec<LlamaToken>,
}

impl LazyGrammarTriggers {
    /// # Errors
    ///
    /// Returns the [`GrammarError`] of escaping a trigger word.
    pub fn of(triggers: Vec<ChatGrammarTrigger>) -> Result<Self, GrammarError> {
        let mut patterns = Vec::new();
        let mut tokens = Vec::new();

        for trigger in triggers {
            match trigger {
                ChatGrammarTrigger::Token(token) => tokens.push(token),
                ChatGrammarTrigger::Word(word) => patterns.push(regex_escaped(&word)?),
                ChatGrammarTrigger::Pattern(pattern) => patterns.push(pattern),
                ChatGrammarTrigger::PatternFull(pattern) => {
                    patterns.push(anchored_pattern(&pattern));
                }
            }
        }

        Ok(Self { patterns, tokens })
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_fault_injection::ArmedFault;
    use llama_cpp_fault_injection::injected_fault_message;

    use super::LazyGrammarTriggers;
    use super::anchored_pattern;
    use super::regex_escape_status_to_result;
    use crate::chat_grammar_trigger::ChatGrammarTrigger;
    use crate::error::grammar_error::GrammarError;
    use crate::token::LlamaToken;

    #[test]
    fn escaping_a_trigger_word_maps_every_status() {
        let escape = |status, out_string, out_error| {
            regex_escape_status_to_result(status, out_string, out_error)
        };
        let contract = |detail| {
            Err(GrammarError::FfiContract(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_regex_escape",
                    detail,
                },
            ))
        };

        assert_eq!(
            [
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_OK,
                    unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"\\[TOOL\\]".as_ptr()) },
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_OK,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    unsafe {
                        llama_cpp_bindings_sys::llama_rs_string_dup(c"the escape threw".as_ptr())
                    }
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_TEXT_ARG,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_OUT_STRING_ARG,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(
                    llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_NULL_OUT_ERROR_ARG,
                    ptr::null_mut(),
                    ptr::null_mut()
                ),
                escape(255, ptr::null_mut(), ptr::null_mut()),
            ],
            [
                Ok("\\[TOOL\\]".to_owned()),
                contract("success status contained a null string"),
                Err(GrammarError::Reported {
                    message: "the escape threw".to_owned()
                }),
                Err(GrammarError::NotEnoughMemory),
                Err(GrammarError::LlamaCppOutOfMemory),
                contract("was given a null text argument"),
                contract("was given a null out_string argument"),
                contract("was given a null out_error argument"),
                Err(GrammarError::FfiStatus(
                    llama_cpp_ffi_status::FfiStatusError {
                        operation: "llama_rs_regex_escape",
                        code: 255
                    }
                )),
            ]
        );
    }

    #[test]
    fn an_escape_failure_without_a_message_is_a_contract_violation() {
        assert_eq!(
            regex_escape_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                ptr::null_mut()
            ),
            Err(GrammarError::FfiContract(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_regex_escape",
                    detail: "reported a thrown C++ exception without an error message"
                }
            ))
        );
    }

    #[test]
    fn a_full_match_pattern_is_anchored_at_both_ends_once() {
        assert_eq!(
            ["", "abc", "^abc", "abc$", "^abc$"].map(anchored_pattern),
            ["^$", "^abc$", "^abc$", "^abc$", "^abc$"].map(str::to_owned)
        );
    }

    #[test]
    fn every_trigger_kind_becomes_a_lazy_grammar_trigger() {
        assert_eq!(
            LazyGrammarTriggers::of(vec![
                ChatGrammarTrigger::Token(LlamaToken::new(151_657)),
                ChatGrammarTrigger::Word("[TOOL_CALLS]".to_owned()),
                ChatGrammarTrigger::Pattern("<tool_call>".to_owned()),
                ChatGrammarTrigger::PatternFull("\\s*\\{".to_owned()),
            ]),
            Ok(LazyGrammarTriggers {
                patterns: vec![
                    "\\[TOOL_CALLS\\]".to_owned(),
                    "<tool_call>".to_owned(),
                    "^\\s*\\{$".to_owned(),
                ],
                tokens: vec![LlamaToken::new(151_657)],
            })
        );
    }

    #[test]
    fn a_trigger_word_llama_cpp_cannot_escape_is_reported() {
        let _fault =
            ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_REGEX_ESCAPE_THROWS);

        assert_eq!(
            LazyGrammarTriggers::of(vec![ChatGrammarTrigger::Word("<tool_call>".to_owned())]),
            Err(GrammarError::Reported {
                message: injected_fault_message(),
            })
        );
    }
}
