use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_types::ModelProvidedToolCallId;
use llama_cpp_bindings_types::ModelProvidedToolCallIdError;
use llama_cpp_bindings_types::ParsedChatMessage;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ParsedToolCallId;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::cpp_text_view::cpp_text_view;
use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::result_after_release::result_after_release;

/// # Safety
///
/// `free_error` must be the pointer populated by the preceding
/// `llama_rs_parsed_chat_free` call, or null. The destructor-threw arm reads and
/// frees it.
unsafe fn parsed_chat_free_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_free_status,
    free_error: *mut c_char,
) -> Result<(), ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    free_error,
                    "llama_rs_parsed_chat_free",
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(ParseChatMessageError::DestructorFailed { message })
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_parsed_chat_free",
            code: i64::from(other),
        }
        .into()),
    }
}

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
    let mut free_error: *mut c_char = ptr::null_mut();
    let free_status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle.as_ptr(), &raw mut free_error)
    };
    let freed = unsafe { parsed_chat_free_status_to_result(free_status, free_error) };

    result_after_release(message, freed)
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_types::ModelProvidedToolCallId;
    use llama_cpp_bindings_types::ParsedToolCallId;

    use super::parsed_chat_free_status_to_result;
    use super::tool_call_id_reported_by_llama_cpp;
    use crate::error::parse_chat_message_error::ParseChatMessageError;

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

    #[test]
    fn parsed_chat_free_ok_is_success() {
        let result = unsafe {
            parsed_chat_free_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_OK,
                ptr::null_mut(),
            )
        };

        assert!(
            result.is_ok(),
            "a clean destructor must not report a failure"
        );
    }

    #[test]
    fn parsed_chat_free_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_free_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
            )
        };

        assert_eq!(result.err(), Some(ParseChatMessageError::NotEnoughMemory));
    }

    #[test]
    fn parsed_chat_free_llama_cpp_out_of_memory_is_preserved() {
        let result = unsafe {
            parsed_chat_free_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result.err(),
            Some(ParseChatMessageError::LlamaCppOutOfMemory)
        );
    }

    #[test]
    fn parsed_chat_free_destructor_threw_surfaces_the_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"the destructor threw".as_ptr())
        };
        let result = unsafe {
            parsed_chat_free_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
                out_error,
            )
        };

        assert_eq!(
            result.err(),
            Some(ParseChatMessageError::DestructorFailed {
                message: "the destructor threw".to_owned()
            })
        );
    }

    #[test]
    fn parsed_chat_free_unknown_status_is_preserved() {
        let result = unsafe { parsed_chat_free_status_to_result(255, ptr::null_mut()) };

        assert_eq!(
            result.err(),
            Some(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_free",
                code: 255,
            }))
        );
    }
}
