use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatGrammarTrigger {
    Token(LlamaToken),
    Word(String),
    Pattern(String),
    PatternFull(String),
}

impl ChatGrammarTrigger {
    /// Builds the trigger the wrapper reports as `trigger_type` with `value`, reading its token
    /// only for a token trigger.
    ///
    /// # Errors
    ///
    /// Returns [`llama_cpp_ffi_status::ffi_status_error::FfiStatusError`] for a trigger type unknown to this binding.
    pub fn from_wrapper(
        trigger_type: llama_cpp_bindings_sys::llama_rs_chat_grammar_trigger_type,
        value: String,
        token: impl FnOnce() -> LlamaToken,
    ) -> Result<Self, FfiStatusError> {
        match trigger_type {
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN => {
                Ok(Self::Token(token()))
            }
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD => {
                Ok(Self::Word(value))
            }
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN => {
                Ok(Self::Pattern(value))
            }
            llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL => {
                Ok(Self::PatternFull(value))
            }
            other => Err(FfiStatusError {
                operation: "llama_rs_chat_session_grammar_trigger_type",
                code: i64::from(other),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use crate::chat_grammar_trigger::ChatGrammarTrigger;
    use crate::token::LlamaToken;

    fn trigger_of(
        trigger_type: llama_cpp_bindings_sys::llama_rs_chat_grammar_trigger_type,
    ) -> Result<ChatGrammarTrigger, FfiStatusError> {
        ChatGrammarTrigger::from_wrapper(trigger_type, "<tool_call>".to_owned(), || {
            LlamaToken::new(151_657)
        })
    }

    #[test]
    fn every_wrapper_trigger_type_maps_to_its_trigger() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN,
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
            ]
            .map(trigger_of),
            [
                Ok(ChatGrammarTrigger::Token(LlamaToken::new(151_657))),
                Ok(ChatGrammarTrigger::Word("<tool_call>".to_owned())),
                Ok(ChatGrammarTrigger::Pattern("<tool_call>".to_owned())),
                Ok(ChatGrammarTrigger::PatternFull("<tool_call>".to_owned())),
            ]
        );
    }

    #[test]
    fn an_unknown_wrapper_trigger_type_is_a_status_error() {
        assert_eq!(
            trigger_of(99),
            Err(FfiStatusError {
                operation: "llama_rs_chat_session_grammar_trigger_type",
                code: 99,
            })
        );
    }
}
