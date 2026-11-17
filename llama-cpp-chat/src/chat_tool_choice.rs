#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatToolChoice {
    Auto,
    None,
    Required,
}

impl From<ChatToolChoice> for llama_cpp_bindings_sys::llama_rs_chat_tool_choice {
    fn from(tool_choice: ChatToolChoice) -> Self {
        match tool_choice {
            ChatToolChoice::Auto => llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOL_CHOICE_AUTO,
            ChatToolChoice::None => llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOL_CHOICE_NONE,
            ChatToolChoice::Required => llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED,
        }
    }
}
