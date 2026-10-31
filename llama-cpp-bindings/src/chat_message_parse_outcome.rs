use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;

use crate::raw_chat_message::RawChatMessage;

#[derive(Debug, Eq, PartialEq)]
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

    /// The message this answer becomes when `tool_calls` were recognized after it: a recognized
    /// answer keeps its content and reasoning, an unrecognized one is kept whole as content.
    #[must_use]
    pub fn followed_by_tool_calls(self, tool_calls: Vec<ParsedToolCall>) -> ParsedChatMessage {
        match self {
            Self::Recognized(message) => {
                ParsedChatMessage::new(message.content, message.reasoning_content, tool_calls)
            }
            Self::Unrecognized(raw) => ParsedChatMessage::new(raw.text, String::new(), tool_calls),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
    use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
    use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;
    use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;

    use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
    use crate::chat_output_completeness::ChatOutputCompleteness;
    use crate::raw_chat_message::RawChatMessage;

    fn weather_call() -> ParsedToolCall {
        ParsedToolCall::new(
            ParsedToolCallId::NotProvidedByModel,
            "get_weather".to_owned(),
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        )
    }

    #[test]
    fn a_recognized_answer_keeps_its_content_and_reasoning_before_the_tool_calls() {
        assert_eq!(
            ChatMessageParseOutcome::Recognized(ParsedChatMessage::new(
                "Checking.".to_owned(),
                "The user wants the weather.".to_owned(),
                Vec::new(),
            ))
            .followed_by_tool_calls(vec![weather_call()]),
            ParsedChatMessage::new(
                "Checking.".to_owned(),
                "The user wants the weather.".to_owned(),
                vec![weather_call()],
            )
        );
    }

    #[test]
    fn an_unrecognized_answer_is_kept_whole_as_content_before_the_tool_calls() {
        assert_eq!(
            ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                text: "<odd>Checking.".to_owned(),
                completeness: ChatOutputCompleteness::Finished,
                ffi_error_message: "the format differs".to_owned(),
            })
            .followed_by_tool_calls(vec![weather_call()]),
            ParsedChatMessage::new(
                "<odd>Checking.".to_owned(),
                String::new(),
                vec![weather_call()]
            )
        );
    }

    #[test]
    fn a_recognized_answer_is_its_message() {
        let message = ParsedChatMessage::new("Paris.".to_owned(), String::new(), Vec::new());

        assert_eq!(
            ChatMessageParseOutcome::Recognized(message.clone()).into_recognized(),
            Ok(message)
        );
    }

    fn odd_answer() -> RawChatMessage {
        RawChatMessage {
            text: "<odd>Paris.".to_owned(),
            completeness: ChatOutputCompleteness::Finished,
            ffi_error_message: "the format differs".to_owned(),
        }
    }

    #[test]
    fn an_unrecognized_answer_is_its_raw_text() {
        assert_eq!(
            ChatMessageParseOutcome::Unrecognized(odd_answer()).into_recognized(),
            Err(odd_answer())
        );
    }
}
