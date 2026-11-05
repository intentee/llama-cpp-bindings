use crate::chat_assistant_prefill::ChatAssistantPrefill;

/// How the rendered prompt leaves the assistant's turn, as llama.cpp's `add_generation_prompt`
/// and `continue_final_message` do.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatAssistantTurn {
    /// The prompt ends after the last message.
    Omitted,
    /// The prompt opens a new assistant turn.
    New,
    /// The prompt opens an assistant turn whose reasoning starts with the given text and is left
    /// open for the model to continue.
    ContinueReasoning(String),
    /// The prompt opens an assistant turn with the given reasoning closed and content left open
    /// for the model to continue.
    ContinueContent(ChatAssistantPrefill),
}

impl From<&ChatAssistantTurn> for llama_cpp_bindings_sys::llama_rs_chat_assistant_turn {
    fn from(assistant_turn: &ChatAssistantTurn) -> Self {
        match assistant_turn {
            ChatAssistantTurn::Omitted => {
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_ASSISTANT_TURN_OMITTED
            }
            ChatAssistantTurn::New => llama_cpp_bindings_sys::LLAMA_RS_CHAT_ASSISTANT_TURN_NEW,
            ChatAssistantTurn::ContinueReasoning(_) => {
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_REASONING
            }
            ChatAssistantTurn::ContinueContent(_) => {
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_CONTENT
            }
        }
    }
}
