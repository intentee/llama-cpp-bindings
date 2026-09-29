use std::mem;

use serde::Deserialize;
use serde::de::IgnoredAny;

use crate::json_probe_outcome::JsonProbeOutcome;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BareJsonToolCall {
    name: String,
    #[serde(rename = "arguments")]
    _arguments: Option<IgnoredAny>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ToolCallField {
    Arguments,
    Name,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeState {
    AwaitingObjectOpen,
    AwaitingFirstKeyOrClose,
    AwaitingKey,
    InKey {
        quoted_key: String,
        escaped: bool,
    },
    AwaitingColon(ToolCallField),
    AwaitingValue(ToolCallField),
    InName {
        escaped: bool,
    },
    InArguments {
        depth: usize,
        in_string: bool,
        escaped: bool,
    },
    AwaitingCommaOrClose,
    Closed,
    Failed,
}

const fn is_json_whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n' | '\r')
}

fn field_named(quoted_key: &str) -> Option<ToolCallField> {
    match serde_json::from_str::<String>(quoted_key).ok()?.as_str() {
        "arguments" => Some(ToolCallField::Arguments),
        "name" => Some(ToolCallField::Name),
        _ => None,
    }
}

fn is_named_tool_call(held_text: &str) -> bool {
    serde_json::from_str::<BareJsonToolCall>(held_text)
        .is_ok_and(|tool_call| !tool_call.name.is_empty())
}

impl ProbeState {
    fn advance(self, character: char) -> Self {
        match self {
            Self::AwaitingObjectOpen => match character {
                '{' => Self::AwaitingFirstKeyOrClose,
                _ if is_json_whitespace(character) => Self::AwaitingObjectOpen,
                _ => Self::Failed,
            },
            Self::AwaitingFirstKeyOrClose => match character {
                '"' => Self::InKey {
                    quoted_key: String::from('"'),
                    escaped: false,
                },
                '}' => Self::Closed,
                _ if is_json_whitespace(character) => Self::AwaitingFirstKeyOrClose,
                _ => Self::Failed,
            },
            Self::AwaitingKey => match character {
                '"' => Self::InKey {
                    quoted_key: String::from('"'),
                    escaped: false,
                },
                _ if is_json_whitespace(character) => Self::AwaitingKey,
                _ => Self::Failed,
            },
            Self::InKey {
                mut quoted_key,
                escaped,
            } => {
                quoted_key.push(character);

                if escaped || character != '"' {
                    Self::InKey {
                        quoted_key,
                        escaped: !escaped && character == '\\',
                    }
                } else {
                    field_named(&quoted_key).map_or(Self::Failed, Self::AwaitingColon)
                }
            }
            Self::AwaitingColon(field) => match character {
                ':' => Self::AwaitingValue(field),
                _ if is_json_whitespace(character) => Self::AwaitingColon(field),
                _ => Self::Failed,
            },
            Self::AwaitingValue(field) => match (field, character) {
                (ToolCallField::Name, '"') => Self::InName { escaped: false },
                (ToolCallField::Arguments, '{') => Self::InArguments {
                    depth: 1,
                    in_string: false,
                    escaped: false,
                },
                _ if is_json_whitespace(character) => Self::AwaitingValue(field),
                _ => Self::Failed,
            },
            Self::InName { escaped } => {
                if !escaped && character == '"' {
                    Self::AwaitingCommaOrClose
                } else {
                    Self::InName {
                        escaped: !escaped && character == '\\',
                    }
                }
            }
            Self::InArguments {
                depth,
                in_string,
                escaped,
            } => Self::advance_arguments(depth, in_string, escaped, character),
            Self::AwaitingCommaOrClose => match character {
                ',' => Self::AwaitingKey,
                '}' => Self::Closed,
                _ if is_json_whitespace(character) => Self::AwaitingCommaOrClose,
                _ => Self::Failed,
            },
            Self::Closed if is_json_whitespace(character) => Self::Closed,
            Self::Closed | Self::Failed => Self::Failed,
        }
    }

    const fn advance_arguments(
        depth: usize,
        in_string: bool,
        escaped: bool,
        character: char,
    ) -> Self {
        if in_string {
            return Self::InArguments {
                depth,
                in_string: escaped || character != '"',
                escaped: !escaped && character == '\\',
            };
        }

        match character {
            '"' => Self::InArguments {
                depth,
                in_string: true,
                escaped: false,
            },
            '{' | '[' => Self::InArguments {
                depth: depth + 1,
                in_string: false,
                escaped: false,
            },
            '}' | ']' if depth == 1 => Self::AwaitingCommaOrClose,
            '}' | ']' => Self::InArguments {
                depth: depth - 1,
                in_string: false,
                escaped: false,
            },
            _ => Self::InArguments {
                depth,
                in_string: false,
                escaped: false,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StreamingJsonProbe {
    held_text: String,
    state: ProbeState,
}

impl Default for StreamingJsonProbe {
    fn default() -> Self {
        Self {
            held_text: String::new(),
            state: ProbeState::AwaitingObjectOpen,
        }
    }
}

impl StreamingJsonProbe {
    pub fn feed(&mut self, piece: &str) -> JsonProbeOutcome {
        self.held_text.push_str(piece);

        for character in piece.chars() {
            let state = mem::replace(&mut self.state, ProbeState::Failed);

            self.state = state.advance(character);

            if self.state == ProbeState::Failed {
                return JsonProbeOutcome::Failed;
            }
        }

        match self.state {
            ProbeState::Closed if is_named_tool_call(&self.held_text) => {
                JsonProbeOutcome::CompletedValid
            }
            ProbeState::Closed | ProbeState::Failed => JsonProbeOutcome::Failed,
            _ => JsonProbeOutcome::StillPossiblyValid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingJsonProbe;
    use crate::json_probe_outcome::JsonProbeOutcome;

    fn probe(buffer: &str) -> JsonProbeOutcome {
        StreamingJsonProbe::default().feed(buffer)
    }

    #[test]
    fn empty_buffer_is_still_possibly_valid() {
        assert_eq!(probe(""), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn whitespace_only_buffer_is_still_possibly_valid() {
        assert_eq!(probe("   \n  "), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn single_open_brace_is_still_possibly_valid() {
        assert_eq!(probe("{"), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn open_brace_with_trailing_space_is_still_possibly_valid() {
        assert_eq!(probe("{ "), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn open_brace_with_quote_starting_key_is_still_possibly_valid() {
        assert_eq!(probe(r#"{ ""#), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn partial_name_key_is_still_possibly_valid() {
        assert_eq!(probe(r#"{ "name""#), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn partial_name_value_quote_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": ""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn partial_name_value_letters_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "ge"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn complete_name_string_no_comma_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_comma_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather","#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_partial_arguments_key_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather", "argum"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_arguments_key_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather", "arguments""#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn name_then_arguments_open_brace_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather", "arguments": {"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn arguments_with_partial_inner_key_value_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather", "arguments": {"location":"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn arguments_with_partial_inner_string_value_is_still_possibly_valid() {
        assert_eq!(
            probe(r#"{ "name": "get_weather", "arguments": {"location": "Pa"#),
            JsonProbeOutcome::StillPossiblyValid,
        );
    }

    #[test]
    fn complete_simple_tool_call_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_internal_whitespace_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name": "f", "arguments": {}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_string_argument_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_multiple_arguments_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"book_flight","arguments":{"from":"NYC","to":"PAR","passengers":2}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_nested_arguments_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{"a":{"b":[1,2,3]}}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_close_brace_inside_string_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{"q":"a } b"}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_escaped_quotes_in_string_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{"q":"he said \"hi\""}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_unicode_strings_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"日本語","arguments":{"city":"パリ"}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_trailing_whitespace_is_completed_valid() {
        assert_eq!(
            probe("{\"name\":\"f\",\"arguments\":{}}\n"),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_with_array_inside_arguments_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{"items":[1,2,3]}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_without_arguments_field_is_completed_valid() {
        assert_eq!(
            probe(r#"{"name":"ping"}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn top_level_array_is_failed() {
        assert_eq!(probe("["), JsonProbeOutcome::Failed);
    }

    #[test]
    fn top_level_scalar_number_is_failed() {
        assert_eq!(probe("123"), JsonProbeOutcome::Failed);
    }

    #[test]
    fn top_level_string_is_failed() {
        assert_eq!(probe(r#""hi""#), JsonProbeOutcome::Failed);
    }

    #[test]
    fn complete_object_with_wrong_first_key_is_failed() {
        assert_eq!(probe(r#"{"foo":"bar"}"#), JsonProbeOutcome::Failed,);
    }

    #[test]
    fn complete_object_with_non_string_name_is_failed() {
        assert_eq!(
            probe(r#"{"name":123,"arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_null_name_is_failed() {
        assert_eq!(
            probe(r#"{"name":null,"arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_arguments_as_array_is_failed() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":[]}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_arguments_as_string_is_failed() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":"hi"}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_third_top_level_key_is_failed() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{},"extra":1}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_empty_name_is_failed() {
        assert_eq!(
            probe(r#"{"name":"","arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn complete_object_with_trailing_garbage_is_failed() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{}}garbage"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn empty_object_is_failed_due_to_missing_required_name() {
        assert_eq!(probe("{}"), JsonProbeOutcome::Failed);
    }

    #[test]
    fn complete_object_with_arguments_only_no_name_is_failed() {
        assert_eq!(probe(r#"{"arguments":{}}"#), JsonProbeOutcome::Failed,);
    }

    #[test]
    fn leading_whitespace_then_open_brace_is_still_possibly_valid() {
        assert_eq!(probe("\n  \n{"), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn leading_whitespace_then_complete_tool_call_is_completed_valid() {
        assert_eq!(
            probe("\n  {\"name\":\"f\",\"arguments\":{}}"),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn complete_tool_call_followed_by_second_object_is_failed() {
        assert_eq!(
            probe(r#"{"name":"a","arguments":{}}{"name":"b","arguments":{}}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn buffer_with_only_open_quote_is_still_possibly_valid() {
        assert_eq!(probe(r#"{ "n"#), JsonProbeOutcome::StillPossiblyValid,);
    }

    #[test]
    fn buffer_with_complete_first_field_unknown_second_key_is_failed() {
        assert_eq!(
            probe(r#"{ "name": "f", "foo": 1}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn unicode_letter_inside_name_value_completes_validly() {
        assert_eq!(
            probe(r#"{"name":"éclair","arguments":{}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn arguments_field_with_explicit_null_is_failed() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":null}"#),
            JsonProbeOutcome::Failed,
        );
    }

    #[test]
    fn syntactically_malformed_object_is_failed() {
        assert_eq!(probe("{,}"), JsonProbeOutcome::Failed,);
    }

    #[test]
    fn key_written_with_an_escape_sequence_is_recognized() {
        assert_eq!(
            probe(r#"{"na\u006de":"f","arguments":{}}"#),
            JsonProbeOutcome::CompletedValid,
        );
    }

    #[test]
    fn tool_call_fed_one_character_at_a_time_completes_on_its_last_character() {
        let tool_call = r#"{"name":"f","arguments":{"q":"a } b"}}"#;
        let mut streaming_probe = StreamingJsonProbe::default();
        let mut outcomes = Vec::new();

        for character in tool_call.chars() {
            outcomes.push(streaming_probe.feed(&character.to_string()));
        }

        let (last_outcome, earlier_outcomes) = outcomes
            .split_last()
            .expect("the tool call must produce outcomes");

        assert_eq!(*last_outcome, JsonProbeOutcome::CompletedValid);
        assert!(
            earlier_outcomes
                .iter()
                .all(|outcome| *outcome == JsonProbeOutcome::StillPossiblyValid)
        );
    }

    #[test]
    fn syntax_error_inside_arguments_fails_when_the_object_closes() {
        assert_eq!(
            probe(r#"{"name":"f","arguments":{"q" 1}}"#),
            JsonProbeOutcome::Failed,
        );
    }
}
