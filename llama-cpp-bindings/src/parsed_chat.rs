use std::ffi::c_char;
use std::ptr::NonNull;

use llama_cpp_bindings_types::model_provided_tool_call_id::ModelProvidedToolCallId;
use llama_cpp_bindings_types::model_provided_tool_call_id_error::ModelProvidedToolCallIdError;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;
use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;

use crate::cpp_text_view::cpp_text_view;
use crate::error::parse_chat_message_error::ParseChatMessageError;

fn parsed_text(
    read: impl FnOnce(*mut usize) -> *const c_char,
) -> Result<String, ParseChatMessageError> {
    let mut length = 0;
    let text = read(&raw mut length);

    unsafe { cpp_text_view(text, length) }
        .map(str::to_owned)
        .map_err(ParseChatMessageError::ParsedTextNotUtf8)
}

fn tool_call_id_reported_by_llama_cpp(llama_cpp_tool_call_id: String) -> ParsedToolCallId {
    match ModelProvidedToolCallId::try_from(llama_cpp_tool_call_id) {
        Ok(model_provided_tool_call_id) => {
            ParsedToolCallId::ModelProvided(model_provided_tool_call_id)
        }
        Err(ModelProvidedToolCallIdError::Empty) => ParsedToolCallId::NotProvidedByModel,
    }
}

fn parsed_tool_call(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<ParsedToolCall, ParseChatMessageError> {
    let id = parsed_text(|out_length| unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id(handle, index, out_length)
    })?;
    let name = parsed_text(|out_length| unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name(handle, index, out_length)
    })?;
    let arguments = parsed_text(|out_length| unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments(handle, index, out_length)
    })?;

    Ok(ParsedToolCall::new(
        tool_call_id_reported_by_llama_cpp(id),
        name,
        ToolCallArguments::from_string(arguments),
    ))
}

fn collect_parsed_chat_message(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    let content = parsed_text(|out_length| unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_content(handle, out_length)
    })?;
    let reasoning_content = parsed_text(|out_length| unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content(handle, out_length)
    })?;
    let tool_call_count =
        unsafe { llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count(handle) };
    let tool_calls = (0..tool_call_count)
        .map(|index| parsed_tool_call(handle, index))
        .collect::<Result<Vec<ParsedToolCall>, ParseChatMessageError>>()?;

    Ok(ParsedChatMessage::new(
        content,
        reasoning_content,
        tool_calls,
    ))
}

/// Reads the message llama.cpp parsed into `handle` and releases the handle.
///
/// # Errors
///
/// Returns [`ParseChatMessageError`] when an accessor fails or reports text that is not UTF-8,
/// or when releasing the handle fails.
pub fn take_parsed_chat_message(
    handle: NonNull<llama_cpp_bindings_sys::llama_rs_parsed_chat>,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    let message = collect_parsed_chat_message(handle.as_ptr());

    unsafe { llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle.as_ptr()) };

    message
}

#[cfg(test)]
mod tests {

    use llama_cpp_bindings_types::model_provided_tool_call_id::ModelProvidedToolCallId;
    use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;

    use super::tool_call_id_reported_by_llama_cpp;

    #[test]
    fn an_empty_llama_cpp_tool_call_id_is_not_provided_by_the_model() {
        assert_eq!(
            tool_call_id_reported_by_llama_cpp(String::new()),
            ParsedToolCallId::NotProvidedByModel
        );
    }

    #[test]
    fn a_non_empty_llama_cpp_tool_call_id_is_provided_by_the_model() {
        assert_eq!(
            Ok(tool_call_id_reported_by_llama_cpp("abc123def".to_owned())),
            ModelProvidedToolCallId::try_from("abc123def".to_owned())
                .map(ParsedToolCallId::ModelProvided)
        );
    }
}
