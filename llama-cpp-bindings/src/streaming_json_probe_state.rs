use crate::bare_json_tool_call_field::BareJsonToolCallField;

const fn is_json_whitespace(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\n' | '\r')
}

fn field_named(quoted_key: &str) -> Option<BareJsonToolCallField> {
    match serde_json::from_str::<String>(quoted_key).ok()?.as_str() {
        "arguments" => Some(BareJsonToolCallField::Arguments),
        "name" => Some(BareJsonToolCallField::Name),
        _ => None,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum StreamingJsonProbeState {
    AwaitingObjectOpen,
    AwaitingFirstKeyOrClose,
    AwaitingKey,
    InKey {
        quoted_key: String,
        escaped: bool,
    },
    AwaitingColon(BareJsonToolCallField),
    AwaitingValue(BareJsonToolCallField),
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

impl StreamingJsonProbeState {
    pub fn advance(self, character: char) -> Self {
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
                (BareJsonToolCallField::Name, '"') => Self::InName { escaped: false },
                (BareJsonToolCallField::Arguments, '{') => Self::InArguments {
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
