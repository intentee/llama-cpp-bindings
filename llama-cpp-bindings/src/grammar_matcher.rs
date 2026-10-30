use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use llguidance::TokenParser;
use llguidance::api::StopReason;

use crate::error::grammar_runtime_error::GrammarRuntimeError;
use crate::mask_outcome::MaskOutcome;

enum StepOutcome<TValue> {
    Produced(TValue),
    BenignStop,
}

fn panic_payload_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_owned();
    }

    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    "the parser panicked with a payload that carries no message".to_owned()
}

fn catch_step_panic<TValue>(
    operation: &'static str,
    step: impl FnOnce() -> TValue,
) -> Result<TValue, GrammarRuntimeError> {
    catch_unwind(AssertUnwindSafe(step)).map_err(|panic_payload| GrammarRuntimeError::Panicked {
        operation,
        message: panic_payload_message(panic_payload.as_ref()),
    })
}

fn stop_reason_to_result(
    stop_reason: StopReason,
    detail: String,
) -> Result<(), GrammarRuntimeError> {
    match stop_reason {
        StopReason::NotStopped
        | StopReason::NoExtension
        | StopReason::NoExtensionBias
        | StopReason::EndOfSentence => Ok(()),
        StopReason::InternalError => {
            Err(GrammarRuntimeError::InternalParserError { message: detail })
        }
        StopReason::LexerTooComplex => {
            Err(GrammarRuntimeError::LexerTooComplex { message: detail })
        }
        StopReason::ParserTooComplex => {
            Err(GrammarRuntimeError::ParserTooComplex { message: detail })
        }
        StopReason::MaxTokensTotal | StopReason::MaxTokensParser => {
            Err(GrammarRuntimeError::MaxTokensReached { message: detail })
        }
    }
}

pub struct GrammarMatcher {
    parser: TokenParser,
}

impl GrammarMatcher {
    #[must_use]
    pub fn new(parser: TokenParser) -> Self {
        let mut parser = parser;
        if parser.is_fresh() {
            parser.start_without_prompt();
        }

        Self { parser }
    }

    #[must_use]
    pub fn deep_clone(&self) -> Self {
        Self {
            parser: self.parser.deep_clone(),
        }
    }

    /// # Errors
    /// Returns [`GrammarRuntimeError`] when the parser reaches a genuine error
    /// state (distinct from a benign grammar completion).
    pub fn compute_mask(&mut self) -> Result<MaskOutcome, GrammarRuntimeError> {
        match self.run("compute_mask", TokenParser::compute_mask)? {
            StepOutcome::Produced(mask) => Ok(MaskOutcome::Constrained(mask)),
            StepOutcome::BenignStop => Ok(MaskOutcome::GrammarComplete),
        }
    }

    /// # Errors
    /// Returns [`GrammarRuntimeError`] when consuming the token drives the parser
    /// into a genuine error state. A token that completes the grammar is a
    /// benign stop and yields `Ok(())`.
    pub fn consume_token(&mut self, token: u32) -> Result<(), GrammarRuntimeError> {
        match self.run("consume_token", |parser| parser.consume_token(token))? {
            StepOutcome::Produced(_) | StepOutcome::BenignStop => Ok(()),
        }
    }

    /// # Errors
    /// Returns [`GrammarRuntimeError`] when the parser cannot be reset.
    pub fn reset(&mut self) -> Result<(), GrammarRuntimeError> {
        match self.run("reset", TokenParser::reset)? {
            StepOutcome::Produced(()) | StepOutcome::BenignStop => Ok(()),
        }
    }

    fn run<TValue, TError: std::fmt::Display>(
        &mut self,
        operation: &'static str,
        step: impl FnOnce(&mut TokenParser) -> Result<TValue, TError>,
    ) -> Result<StepOutcome<TValue>, GrammarRuntimeError> {
        match catch_step_panic(operation, || step(&mut self.parser))? {
            Ok(value) => Ok(StepOutcome::Produced(value)),
            Err(step_error) => {
                let detail = self
                    .parser
                    .error_message()
                    .unwrap_or_else(|| step_error.to_string());
                stop_reason_to_result(self.parser.stop_reason(), detail)?;

                Ok(StepOutcome::BenignStop)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use llguidance::ParserFactory;
    use llguidance::TokenParser;
    use llguidance::api::StopReason;
    use llguidance::api::TopLevelGrammar;
    use toktrie::ApproximateTokEnv;

    use super::GrammarMatcher;
    use super::catch_step_panic;
    use super::panic_payload_message;
    use super::stop_reason_to_result;
    use crate::error::grammar_runtime_error::GrammarRuntimeError;

    fn letters_a_then_b_parser() -> TokenParser {
        ParserFactory::new_simple(&ApproximateTokEnv::single_byte_env())
            .unwrap()
            .create_parser(TopLevelGrammar::from_regex("ab"))
            .unwrap()
    }

    #[test]
    fn a_started_parser_keeps_the_tokens_it_consumed() {
        let mut parser = letters_a_then_b_parser();

        parser.start_without_prompt();
        parser.consume_token(u32::from(b'a')).unwrap();

        assert_eq!(
            GrammarMatcher::new(parser).consume_token(u32::from(b'b')),
            Ok(())
        );
    }

    #[test]
    fn a_panicking_parser_step_is_reported_with_its_operation() {
        let mut matcher = GrammarMatcher::new(letters_a_then_b_parser());

        assert_eq!(
            matcher
                .run("compute_mask", |_parser| -> Result<(), String> {
                    panic!("parser exploded")
                })
                .err(),
            Some(GrammarRuntimeError::Panicked {
                operation: "compute_mask",
                message: "parser exploded".to_owned(),
            })
        );
    }

    #[test]
    fn a_static_str_panic_payload_is_preserved() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("parser exploded");

        assert_eq!(panic_payload_message(payload.as_ref()), "parser exploded");
    }

    #[test]
    fn an_owned_string_panic_payload_is_preserved() {
        let payload: Box<dyn std::any::Any + Send> = Box::new("lexer exploded".to_owned());

        assert_eq!(panic_payload_message(payload.as_ref()), "lexer exploded");
    }

    #[test]
    fn a_payload_without_a_message_is_described() {
        let payload: Box<dyn std::any::Any + Send> = Box::new(42_u32);

        assert_eq!(
            panic_payload_message(payload.as_ref()),
            "the parser panicked with a payload that carries no message"
        );
    }

    #[test]
    fn a_panicking_step_is_reported_with_its_operation_and_message() {
        assert_eq!(
            catch_step_panic("consume_token", || -> u32 { panic!("parser exploded") }),
            Err(GrammarRuntimeError::Panicked {
                operation: "consume_token",
                message: "parser exploded".to_owned(),
            })
        );
    }

    #[test]
    fn benign_stop_reasons_are_ok() {
        for reason in [
            StopReason::NotStopped,
            StopReason::NoExtension,
            StopReason::NoExtensionBias,
            StopReason::EndOfSentence,
        ] {
            assert!(stop_reason_to_result(reason, String::new()).is_ok());
        }
    }

    #[test]
    fn internal_error_maps_to_internal_parser_error_with_message() {
        assert_eq!(
            stop_reason_to_result(StopReason::InternalError, "boom".to_string()),
            Err(GrammarRuntimeError::InternalParserError {
                message: "boom".to_string()
            })
        );
    }

    #[test]
    fn lexer_too_complex_maps_to_lexer_too_complex() {
        assert_eq!(
            stop_reason_to_result(StopReason::LexerTooComplex, String::new()),
            Err(GrammarRuntimeError::LexerTooComplex {
                message: String::new()
            })
        );
    }

    #[test]
    fn parser_too_complex_maps_to_parser_too_complex() {
        assert_eq!(
            stop_reason_to_result(StopReason::ParserTooComplex, String::new()),
            Err(GrammarRuntimeError::ParserTooComplex {
                message: String::new()
            })
        );
    }

    #[test]
    fn max_token_stop_reasons_map_to_max_tokens_reached() {
        for reason in [StopReason::MaxTokensTotal, StopReason::MaxTokensParser] {
            assert_eq!(
                stop_reason_to_result(reason, String::new()),
                Err(GrammarRuntimeError::MaxTokensReached {
                    message: String::new()
                })
            );
        }
    }
}
