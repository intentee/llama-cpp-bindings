use crate::chat_assistant_turn::ChatAssistantTurn;
use crate::chat_tool_choice::ChatToolChoice;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatSessionOptions {
    pub assistant_turn: ChatAssistantTurn,
    pub enable_thinking: bool,
    pub parallel_tool_calls: bool,
    pub tool_choice: ChatToolChoice,
}

impl From<&ChatSessionOptions> for llama_cpp_bindings_sys::llama_rs_chat_session_options {
    fn from(
        ChatSessionOptions {
            assistant_turn,
            enable_thinking,
            parallel_tool_calls,
            tool_choice,
        }: &ChatSessionOptions,
    ) -> Self {
        Self {
            assistant_turn: assistant_turn.into(),
            enable_thinking: *enable_thinking,
            parallel_tool_calls: *parallel_tool_calls,
            tool_choice: (*tool_choice).into(),
        }
    }
}
