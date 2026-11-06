use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;

use crate::raw_chat_message::RawChatMessage;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessageParseOutcome {
    Recognized(ParsedChatMessage),
    Unrecognized(RawChatMessage),
}

impl ChatMessageParseOutcome {
    /// The recognized message, or the raw answer the parser did not recognize.
    ///
    /// # Errors
    ///
    /// Returns the [`RawChatMessage`] when the answer does not match the template's format.
    pub fn into_recognized(self) -> Result<ParsedChatMessage, RawChatMessage> {
        match self {
            Self::Recognized(message) => Ok(message),
            Self::Unrecognized(raw) => Err(raw),
        }
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;

    use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::raw_chat_message::RawChatMessage;

    fn odd_answer() -> RawChatMessage {
        RawChatMessage {
            text: "<odd>Paris.".to_owned(),
            completeness: ChatOutputCompleteness::Finished,
            expected_format: "peg-native".to_owned(),
        }
    }

    #[test]
    fn a_recognized_answer_is_its_message() {
        let message = ParsedChatMessage::new("Paris.".to_owned(), String::new(), Vec::new());

        assert_eq!(
            ChatMessageParseOutcome::Recognized(message.clone()).into_recognized(),
            Ok(message)
        );
    }

    #[test]
    fn an_unrecognized_answer_is_its_raw_message() {
        assert_eq!(
            ChatMessageParseOutcome::Unrecognized(odd_answer()).into_recognized(),
            Err(odd_answer())
        );
    }
}
