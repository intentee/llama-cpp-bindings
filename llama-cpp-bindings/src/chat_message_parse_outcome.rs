use llama_cpp_bindings_types::ParsedChatMessage;

use crate::raw_chat_message::RawChatMessage;

#[derive(Debug, Eq, PartialEq)]
pub enum ChatMessageParseOutcome {
    Recognized(ParsedChatMessage),
    Unrecognized(RawChatMessage),
}
