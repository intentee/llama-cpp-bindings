use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;

use crate::raw_chat_message::RawChatMessage;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatMessageParseOutcome {
    Recognized(ParsedChatMessage),
    Unrecognized(RawChatMessage),
}

impl ChatMessageParseOutcome {
    /// The recognized message, or `None` for an answer the parser did not recognize.
    #[must_use]
    pub fn into_recognized(self) -> Option<ParsedChatMessage> {
        match self {
            Self::Recognized(message) => Some(message),
            Self::Unrecognized(_) => None,
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
            Some(message)
        );
    }

    #[test]
    fn an_unrecognized_answer_has_no_recognized_message() {
        assert_eq!(
            ChatMessageParseOutcome::Unrecognized(odd_answer()).into_recognized(),
            None
        );
    }
}
