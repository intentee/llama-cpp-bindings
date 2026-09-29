use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_types::ParsedChatMessage;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_bindings_types::ReasoningMarkers;
use llama_cpp_bindings_types::ToolCallArguments;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
use crate::chat_template_tool_calls;
use crate::error::parse_chat_message_error::ParseChatMessageError;
use crate::model::LlamaModel;
use crate::raw_chat_message::RawChatMessage;
use crate::tool_call_format;
use crate::tool_call_format::ToolCallFormatOutcome;

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

/// # Safety
///
/// `out_error` must reference the pointer populated by a `llama_rs_parse_chat_message` call
/// that reported a thrown C++ exception. The error is read, freed, and the referenced pointer
/// is nulled so the later free in the caller does not double-free.
unsafe fn thrown_parse_exception_error(
    out_error: *mut *mut c_char,
    error_from_message: fn(String) -> ParseChatMessageError,
) -> ParseChatMessageError {
    match unsafe {
        read_and_free_cpp_string(
            *out_error,
            "llama_rs_parse_chat_message",
            "reported a thrown C++ exception without an error message",
        )
    } {
        Ok(message) => {
            unsafe { *out_error = ptr::null_mut() };

            error_from_message(message)
        }
        Err(missing_message) => missing_message.into(),
    }
}

/// # Safety
///
/// `handle` must be the parsed-chat handle (or null) and `out_error` must reference the
/// pointer populated by the preceding `llama_rs_parse_chat_message` call. In the CXX-exception
/// arms the error is read, freed, and the referenced pointer is nulled so the later free in the
/// caller does not double-free.
unsafe fn parse_chat_message_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parse_chat_message_status,
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    out_error: *mut *mut c_char,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_OK => {
            if handle.is_null() {
                Err(crate::FfiContractError {
                    operation: "llama_rs_parse_chat_message",
                    detail: "success status contained a null parsed-chat handle",
                }
                .into())
            } else {
                collect_parsed_chat_message(handle)
            }
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe {
                thrown_parse_exception_error(out_error, |message| {
                    ParseChatMessageError::MessageUnrecognized { message }
                })
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_TOOLS_PARSER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null tools parser argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null input argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null out_handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_parse_chat_message",
            code: i64::from(other),
        }
        .into()),
    }
}

fn outcome_from_via_ffi_result(
    via_ffi_result: Result<ParsedChatMessage, ParseChatMessageError>,
    input: &str,
    is_partial: bool,
) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
    match via_ffi_result {
        Ok(mut parsed) => {
            synthesize_missing_tool_call_ids(&mut parsed.tool_calls);
            Ok(ChatMessageParseOutcome::Recognized(parsed))
        }
        Err(ParseChatMessageError::MessageUnrecognized { message }) => {
            Ok(ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                text: input.to_owned(),
                is_partial,
                ffi_error_message: message,
            }))
        }
        Err(other) => Err(other),
    }
}

fn collect_parsed_chat_message(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<ParsedChatMessage, ParseChatMessageError> {
    if handle.is_null() {
        return Ok(ParsedChatMessage::default());
    }

    let content = read_parsed_chat_content(handle)?;
    let reasoning_content = read_parsed_chat_reasoning_content(handle)?;
    let count = read_parsed_chat_tool_call_count(handle)?;

    let mut tool_calls = Vec::with_capacity(count);
    for index in 0..count {
        let id = read_parsed_chat_tool_call_id(handle, index)?;
        let name = read_parsed_chat_tool_call_name(handle, index)?;
        let arguments_json = read_parsed_chat_tool_call_arguments(handle, index)?;

        let arguments = ToolCallArguments::from_string(arguments_json);
        tool_calls.push(ParsedToolCall::new(id, name, arguments));
    }

    Ok(ParsedChatMessage::new(
        content,
        reasoning_content,
        tool_calls,
    ))
}

/// # Safety
///
/// `out_string` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_parsed_chat_content` call (or null when no value/error was produced); each is
/// read and freed in exactly one match arm.
unsafe fn parsed_chat_content_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_content_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_OK => {
            consume_accessor_string(out_string, "llama_rs_parsed_chat_content")
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_parsed_chat_content",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_content",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_content",
                detail: "was given a null out_string argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_content",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_content(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_content(
            handle,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    unsafe { parsed_chat_content_status_to_result(status, out_string, out_error) }
}

/// # Safety
///
/// `out_string` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_parsed_chat_reasoning_content` call (or null when no value/error was produced);
/// each is read and freed in exactly one match arm.
unsafe fn parsed_chat_reasoning_content_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content_status,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK => {
            consume_accessor_string(out_string, "llama_rs_parsed_chat_reasoning_content")
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_and_free_cpp_string(out_error, "llama_rs_parsed_chat_reasoning_content", "reported a thrown C++ exception without an error message") }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                detail: "was given a null out_string argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_reasoning_content(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_reasoning_content(
            handle,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    unsafe { parsed_chat_reasoning_content_status_to_result(status, out_string, out_error) }
}

/// # Safety
///
/// `out_error` must be the pointer populated by the preceding
/// `llama_rs_parsed_chat_tool_call_count` call (or null when no error was produced); it is
/// freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_count_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count_status,
    out_count: usize,
    out_error: *mut c_char,
) -> Result<usize, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK => Ok(out_count),
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_and_free_cpp_string(out_error, "llama_rs_parsed_chat_tool_call_count", "reported a thrown C++ exception without an error message") }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                detail: "was given a null out_count argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_tool_call_count(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
) -> Result<usize, ParseChatMessageError> {
    let mut out_count: usize = 0;
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_count(
            handle,
            &raw mut out_count,
            &raw mut out_error,
        )
    };
    unsafe { parsed_chat_tool_call_count_status_to_result(status, out_count, out_error) }
}

/// # Safety
///
/// `out_string` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_parsed_chat_tool_call_id` call (or null when no value/error was produced); each
/// is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_id_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK => {
            consume_accessor_string(out_string, "llama_rs_parsed_chat_tool_call_id")
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallIdIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_and_free_cpp_string(out_error, "llama_rs_parsed_chat_tool_call_id", "reported a thrown C++ exception without an error message") }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                detail: "was given a null out_string argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_tool_call_id(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_id(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    unsafe { parsed_chat_tool_call_id_status_to_result(status, index, out_string, out_error) }
}

/// # Safety
///
/// `out_string` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_parsed_chat_tool_call_name` call (or null when no value/error was produced); each
/// is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_name_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK => {
            consume_accessor_string(out_string, "llama_rs_parsed_chat_tool_call_name")
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallNameIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_and_free_cpp_string(out_error, "llama_rs_parsed_chat_tool_call_name", "reported a thrown C++ exception without an error message") }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                detail: "was given a null out_string argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_tool_call_name(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_name(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    unsafe { parsed_chat_tool_call_name_status_to_result(status, index, out_string, out_error) }
}

/// # Safety
///
/// `out_string` and `out_error` must be the pointers populated by the preceding
/// `llama_rs_parsed_chat_tool_call_arguments` call (or null when no value/error was produced);
/// each is read and freed in exactly one match arm.
unsafe fn parsed_chat_tool_call_arguments_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments_status,
    index: usize,
    out_string: *mut c_char,
    out_error: *mut c_char,
) -> Result<String, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK => {
            consume_accessor_string(out_string, "llama_rs_parsed_chat_tool_call_arguments")
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS => {
            Err(ParseChatMessageError::ToolCallArgumentsIndexOutOfBounds { index })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_LLAMA_CPP_OUT_OF_MEMORY => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_and_free_cpp_string(out_error, "llama_rs_parsed_chat_tool_call_arguments", "reported a thrown C++ exception without an error message") }?;
            Err(ParseChatMessageError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                detail: "was given a null handle argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                detail: "was given a null out_string argument",
            }
            .into())
        }
        other => {
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_string) };
            unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };
            Err(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                code: i64::from(other),
            }
            .into())
        }
    }
}

fn read_parsed_chat_tool_call_arguments(
    handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat,
    index: usize,
) -> Result<String, ParseChatMessageError> {
    let mut out_string: *mut c_char = ptr::null_mut();
    let mut out_error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_parsed_chat_tool_call_arguments(
            handle,
            index,
            &raw mut out_string,
            &raw mut out_error,
        )
    };
    unsafe {
        parsed_chat_tool_call_arguments_status_to_result(status, index, out_string, out_error)
    }
}

fn consume_accessor_string(
    ptr: *mut c_char,
    operation: &'static str,
) -> Result<String, ParseChatMessageError> {
    if ptr.is_null() {
        return Err(crate::FfiContractError {
            operation,
            detail: "success status contained a null string",
        }
        .into());
    }
    let bytes = unsafe { CStr::from_ptr(ptr) }.to_bytes().to_vec();
    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(ptr) };
    Ok(String::from_utf8(bytes)?)
}

struct ReasoningSplit {
    reasoning: String,
    content: String,
}

fn restore_partial_reasoning(
    parsed: &mut ParsedChatMessage,
    input: &str,
    reasoning_markers: Option<&ReasoningMarkers>,
    is_partial: bool,
) {
    if !is_partial {
        return;
    }
    if reasoning_markers.is_some_and(|markers| input.contains(&markers.open)) {
        let split = split_reasoning_prefix(input, reasoning_markers, None, true);
        parsed.reasoning_content = split.reasoning;
        parsed.content = split.content;
        return;
    }
    if let Some(open) = reasoning_markers.map(|markers| markers.open.trim())
        && let Some(reasoning) = parsed.reasoning_content.trim_start().strip_prefix(open)
    {
        parsed.reasoning_content = reasoning.to_owned();
    }
}

fn split_reasoning_prefix(
    input: &str,
    reasoning_markers: Option<&ReasoningMarkers>,
    tool_call_open: Option<&str>,
    is_partial: bool,
) -> ReasoningSplit {
    let content_only = || ReasoningSplit {
        reasoning: String::new(),
        content: prefix_before_optional(input, tool_call_open),
    };

    let Some(reasoning_markers) = reasoning_markers else {
        return content_only();
    };
    let Some(open_pos) = input.find(&reasoning_markers.open) else {
        return content_only();
    };

    let after_open = &input[open_pos + reasoning_markers.open.len()..];
    let closing_marker = reasoning_markers
        .closes
        .iter()
        .enumerate()
        .filter_map(|(marker_index, marker)| {
            after_open
                .find(marker)
                .map(|offset| (offset, marker_index, marker))
        })
        .min_by_key(|(offset, marker_index, _)| (*offset, *marker_index));
    let Some((close_offset, _, close_marker)) = closing_marker else {
        return if is_partial {
            ReasoningSplit {
                reasoning: prefix_before_optional(after_open, tool_call_open),
                content: input[..open_pos].to_owned(),
            }
        } else {
            content_only()
        };
    };

    let reasoning = after_open[..close_offset].to_owned();
    let after_close = &after_open[close_offset + close_marker.len()..];

    ReasoningSplit {
        reasoning,
        content: prefix_before_optional(after_close, tool_call_open),
    }
}

fn prefix_before_optional(text: &str, marker: Option<&str>) -> String {
    marker.map_or_else(
        || text.to_owned(),
        |marker| {
            text.find(marker)
                .map_or_else(|| text.to_owned(), |pos| text[..pos].to_owned())
        },
    )
}

fn synthesize_missing_tool_call_ids(tool_calls: &mut [ParsedToolCall]) {
    for (index, call) in tool_calls.iter_mut().enumerate() {
        if call.id.is_empty() {
            call.id = format!("call_{index}");
        }
    }
}

/// # Safety
///
/// `out_error` must reference the pointer populated by the preceding
/// `llama_rs_chat_tools_parser_create` call (or null); it is read, freed, and nulled only in
/// the CXX-exception arm. `tools_parser` must be the pointer populated by the same call.
unsafe fn chat_tools_parser_create_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_tools_parser_create_status,
    tools_parser: *mut llama_cpp_bindings_sys::llama_rs_chat_tools_parser,
    out_error: *mut *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::llama_rs_chat_tools_parser>, ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_OK => NonNull::new(tools_parser)
            .ok_or_else(|| {
                crate::FfiContractError {
                    operation: "llama_rs_chat_tools_parser_create",
                    detail: "success status contained a null tools parser handle",
                }
                .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_TOOLS_NOT_AN_ARRAY => {
            Err(ParseChatMessageError::ToolsNotAnArray)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    *out_error,
                    "llama_rs_chat_tools_parser_create",
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            unsafe { *out_error = ptr::null_mut() };

            Err(ParseChatMessageError::ToolsParserBuildFailed { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_PARSER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "was given a null parser argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_TOOLS_JSON_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "was given a null tools_json argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_OUT_TOOLS_PARSER_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "was given a null out_tools_parser argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_OUT_ERROR_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "was given a null out_error argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_chat_tools_parser_create",
            code: i64::from(other),
        }
        .into()),
    }
}

/// # Safety
///
/// `out_error` must be the pointer populated by the preceding
/// `llama_rs_chat_tools_parser_free` call, or null. The destructor-threw arm reads and frees it.
unsafe fn chat_tools_parser_free_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_chat_tools_parser_free_status,
    out_error: *mut c_char,
) -> Result<(), ParseChatMessageError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(ParseChatMessageError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_chat_tools_parser_free",
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(ParseChatMessageError::DestructorFailed { message })
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_chat_tools_parser_free",
            code: i64::from(other),
        }
        .into()),
    }
}

/// Parses generated chat messages against one set of tools, building the
/// model-specific tools parser once for every message it parses.
pub struct ChatMessageParser {
    reasoning_markers: Option<ReasoningMarkers>,
    tools_parser: NonNull<llama_cpp_bindings_sys::llama_rs_chat_tools_parser>,
}

unsafe impl Send for ChatMessageParser {}

impl ChatMessageParser {
    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when reasoning-marker detection fails, the model has
    /// no chat parser, `tools_json` contains a NUL byte or is not a JSON array, or the tools
    /// parser cannot be built.
    pub fn new(model: &LlamaModel, tools_json: &str) -> Result<Self, ParseChatMessageError> {
        let reasoning_markers = model.reasoning_markers()?.cloned();
        let chat_parser = model.chat_parser_ptr()?;
        let tools_json_cstring =
            CString::new(tools_json).map_err(ParseChatMessageError::ToolsContainNulByte)?;
        let mut out_tools_parser: *mut llama_cpp_bindings_sys::llama_rs_chat_tools_parser =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_tools_parser_create(
                chat_parser,
                tools_json_cstring.as_ptr(),
                &raw mut out_tools_parser,
                &raw mut out_error,
            )
        };

        let tools_parser = unsafe {
            chat_tools_parser_create_status_to_result(status, out_tools_parser, &raw mut out_error)
        };

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };

        Ok(Self {
            reasoning_markers,
            tools_parser: tools_parser?,
        })
    }

    /// # Errors
    ///
    /// Returns [`ParseChatMessageError`] when `input` contains a NUL byte, the FFI returns a
    /// non-OK status other than a message parse exception, or accessor strings are not valid
    /// UTF-8.
    pub fn parse(
        &self,
        input: &str,
        is_partial: bool,
    ) -> Result<ChatMessageParseOutcome, ParseChatMessageError> {
        let reasoning_markers = self.reasoning_markers.as_ref();

        for candidate in chat_template_tool_calls::known_marker_candidates() {
            match tool_call_format::try_parse(input, &candidate) {
                ToolCallFormatOutcome::NoMatch => {}
                ToolCallFormatOutcome::Parsed(calls) => {
                    let split = split_reasoning_prefix(
                        input,
                        reasoning_markers,
                        Some(&candidate.open),
                        is_partial,
                    );
                    let mut parsed = ParsedChatMessage::new(split.content, split.reasoning, calls);
                    synthesize_missing_tool_call_ids(&mut parsed.tool_calls);

                    return Ok(ChatMessageParseOutcome::Recognized(parsed));
                }
                ToolCallFormatOutcome::Failed(_shape_does_not_fit) => {}
            }
        }

        let via_ffi_result = self.parse_via_ffi(input, is_partial).map(|mut parsed| {
            restore_partial_reasoning(&mut parsed, input, reasoning_markers, is_partial);
            parsed
        });

        outcome_from_via_ffi_result(via_ffi_result, input, is_partial)
    }

    fn parse_via_ffi(
        &self,
        input: &str,
        is_partial: bool,
    ) -> Result<ParsedChatMessage, ParseChatMessageError> {
        let input_cstring =
            CString::new(input).map_err(ParseChatMessageError::InputContainsNulByte)?;

        let mut handle: *mut llama_cpp_bindings_sys::llama_rs_parsed_chat = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_parse_chat_message(
                self.tools_parser.as_ptr(),
                input_cstring.as_ptr(),
                i32::from(is_partial),
                &raw mut handle,
                &raw mut out_error,
            )
        };

        let parsed =
            unsafe { parse_chat_message_status_to_result(status, handle, &raw mut out_error) };

        let mut free_error: *mut c_char = ptr::null_mut();
        let free_status = unsafe {
            llama_cpp_bindings_sys::llama_rs_parsed_chat_free(handle, &raw mut free_error)
        };
        let freed = unsafe { parsed_chat_free_status_to_result(free_status, free_error) };

        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_error) };

        match parsed {
            Ok(message) => freed.map(|()| message),
            Err(parse_failure) => {
                if let Err(destructor_failure) = freed {
                    log::error!("{destructor_failure}");
                }

                Err(parse_failure)
            }
        }
    }
}

impl Drop for ChatMessageParser {
    fn drop(&mut self) {
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_tools_parser_free(
                self.tools_parser.as_ptr(),
                &raw mut out_error,
            )
        };

        if let Err(destructor_failure) =
            unsafe { chat_tools_parser_free_status_to_result(status, out_error) }
        {
            log::error!("{destructor_failure}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;
    use std::ffi::c_char;
    use std::mem::discriminant;
    use std::ptr;

    use llama_cpp_bindings_types::ParsedChatMessage;
    use llama_cpp_bindings_types::ParsedToolCall;
    use llama_cpp_bindings_types::ReasoningMarkers;
    use llama_cpp_bindings_types::ToolCallArguments;

    use super::ReasoningSplit;
    use super::chat_tools_parser_create_status_to_result;
    use super::chat_tools_parser_free_status_to_result;
    use super::outcome_from_via_ffi_result;
    use super::parse_chat_message_status_to_result;
    use super::parsed_chat_content_status_to_result;
    use super::parsed_chat_free_status_to_result;
    use super::parsed_chat_reasoning_content_status_to_result;
    use super::parsed_chat_tool_call_arguments_status_to_result;
    use super::parsed_chat_tool_call_count_status_to_result;
    use super::parsed_chat_tool_call_id_status_to_result;
    use super::parsed_chat_tool_call_name_status_to_result;
    use super::restore_partial_reasoning;
    use super::split_reasoning_prefix;
    use crate::chat_message_parse_outcome::ChatMessageParseOutcome;
    use crate::error::parse_chat_message_error::ParseChatMessageError;
    use crate::raw_chat_message::RawChatMessage;
    #[test]
    fn parse_chat_message_success_with_null_handle_is_contract_error() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_OK,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parse_chat_message",
                    detail: "success status contained a null parsed-chat handle",
                }
            ))
        ));
    }

    #[test]
    fn parse_chat_message_allocation_failed_is_not_enough_memory() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parse_chat_message_cxx_exception_is_message_unrecognized_and_nulls_error() {
        let mut out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"the message could not be parsed".as_ptr())
        };
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        let Err(ParseChatMessageError::MessageUnrecognized { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "the message could not be parsed");
        assert!(
            out_error.is_null(),
            "the reclaimed pointer must be nulled so the caller does not free it twice"
        );
    }

    #[test]
    fn parse_chat_message_cxx_exception_without_an_error_message_is_a_contract_error() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                &raw mut out_error,
            )
        };

        assert_eq!(
            result.unwrap_err(),
            ParseChatMessageError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "reported a thrown C++ exception without an error message",
            })
        );
    }

    #[test]
    fn parse_chat_message_unknown_status_is_preserved() {
        let mut out_error: *mut c_char = ptr::null_mut();
        let result = unsafe {
            parse_chat_message_status_to_result(255, ptr::null_mut(), &raw mut out_error)
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parse_chat_message",
                code: 255,
            }))
        );
    }

    #[test]
    fn parsed_chat_content_success_with_null_string_is_contract_error() {
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parsed_chat_content",
                    detail: "success status contained a null string",
                }
            ))
        ));
    }

    #[test]
    fn parsed_chat_content_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_content_cxx_exception_is_reported() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"content read failed".as_ptr()) };
        let result = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "content read failed");
    }

    #[test]
    fn parsed_chat_content_unknown_status_is_preserved() {
        let result =
            unsafe { parsed_chat_content_status_to_result(255, ptr::null_mut(), ptr::null_mut()) };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_content",
                code: 255,
            }))
        ));
    }

    #[test]
    fn parsed_chat_reasoning_content_success_with_null_string_is_contract_error() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_OK,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parsed_chat_reasoning_content",
                    detail: "success status contained a null string",
                }
            ))
        ));
    }

    #[test]
    fn parsed_chat_reasoning_content_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_reasoning_content_cxx_exception_is_reported() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"reasoning read failed".as_ptr())
        };
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "reasoning read failed");
    }

    #[test]
    fn parsed_chat_reasoning_content_unknown_status_is_preserved() {
        let result = unsafe {
            parsed_chat_reasoning_content_status_to_result(255, ptr::null_mut(), ptr::null_mut())
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                code: 255,
            }))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_count_ok_returns_count() {
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_OK,
                7,
                ptr::null_mut(),
            )
        };

        assert_eq!(result.unwrap(), 7);
    }

    #[test]
    fn parsed_chat_tool_call_count_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_count_cxx_exception_is_reported() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"tool-call count failed".as_ptr())
        };
        let result = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "tool-call count failed");
    }

    #[test]
    fn parsed_chat_tool_call_count_unknown_status_is_preserved() {
        let result =
            unsafe { parsed_chat_tool_call_count_status_to_result(255, 0, ptr::null_mut()) };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                code: 255,
            }))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_id_success_with_null_string_is_contract_error() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parsed_chat_tool_call_id",
                    detail: "success status contained a null string",
                }
            ))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_id_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_INDEX_OUT_OF_BOUNDS,
                4,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallIdIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallIdIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 4);
    }

    #[test]
    fn parsed_chat_tool_call_id_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_id_cxx_exception_is_reported() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"tool-call id read failed".as_ptr())
        };
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "tool-call id read failed");
    }

    #[test]
    fn parsed_chat_tool_call_id_unknown_status_is_preserved() {
        let result = unsafe {
            parsed_chat_tool_call_id_status_to_result(255, 0, ptr::null_mut(), ptr::null_mut())
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                code: 255,
            }))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_name_success_with_null_string_is_contract_error() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parsed_chat_tool_call_name",
                    detail: "success status contained a null string",
                }
            ))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_name_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_INDEX_OUT_OF_BOUNDS,
                2,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallNameIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallNameIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 2);
    }

    #[test]
    fn parsed_chat_tool_call_name_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_name_cxx_exception_is_reported() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"tool-call name read failed".as_ptr())
        };
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "tool-call name read failed");
    }

    #[test]
    fn parsed_chat_tool_call_name_unknown_status_is_preserved() {
        let result = unsafe {
            parsed_chat_tool_call_name_status_to_result(255, 0, ptr::null_mut(), ptr::null_mut())
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                code: 255,
            }))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_arguments_success_with_null_string_is_contract_error() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_OK,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiContract(
                crate::FfiContractError {
                    operation: "llama_rs_parsed_chat_tool_call_arguments",
                    detail: "success status contained a null string",
                }
            ))
        ));
    }

    #[test]
    fn parsed_chat_tool_call_arguments_out_of_bounds_carries_index() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_INDEX_OUT_OF_BOUNDS,
                9,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::ToolCallArgumentsIndexOutOfBounds { index }) = result else {
            panic!("expected ToolCallArgumentsIndexOutOfBounds, got {result:?}");
        };
        assert_eq!(index, 9);
    }

    #[test]
    fn parsed_chat_tool_call_arguments_allocation_failed_is_not_enough_memory() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(
            discriminant(&result.unwrap_err()),
            discriminant(&ParseChatMessageError::NotEnoughMemory)
        );
    }

    #[test]
    fn parsed_chat_tool_call_arguments_cxx_exception_is_reported() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(c"tool-call arguments read failed".as_ptr())
        };
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
                out_error,
            )
        };

        let Err(ParseChatMessageError::Reported { message }) = result else {
            panic!("the llama.cpp exception status must surface the wrapper message");
        };

        assert_eq!(message, "tool-call arguments read failed");
    }

    #[test]
    fn parsed_chat_tool_call_arguments_unknown_status_is_preserved() {
        let result = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                255,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert!(matches!(
            result,
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                code: 255,
            }))
        ));
    }

    #[test]
    fn split_reasoning_prefix_without_markers_returns_content_up_to_tool_call_open() {
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("answer<tool>rest", None, Some("<tool>"), false);

        assert!(reasoning.is_empty());
        assert_eq!(content, "answer");
    }

    #[test]
    fn split_reasoning_prefix_with_missing_open_marker_returns_content_only() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("plain answer", Some(&markers), Some("<tool>"), false);

        assert!(reasoning.is_empty());
        assert_eq!(content, "plain answer");
    }

    #[test]
    fn split_reasoning_prefix_with_missing_close_marker_returns_content_only() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("<think>unterminated", Some(&markers), Some("<tool>"), false);

        assert!(reasoning.is_empty());
        assert_eq!(content, "<think>unterminated");
    }

    #[test]
    fn split_reasoning_prefix_with_partial_unclosed_marker_returns_reasoning() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let ReasoningSplit { reasoning, content } = split_reasoning_prefix(
            "prefix<think>unfinished<tool>tail",
            Some(&markers),
            Some("<tool>"),
            true,
        );

        assert_eq!(reasoning, "unfinished");
        assert_eq!(content, "prefix");
    }

    #[test]
    fn split_reasoning_prefix_without_tool_marker_preserves_all_partial_reasoning() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let ReasoningSplit { reasoning, content } =
            split_reasoning_prefix("<think>unfinished", Some(&markers), None, true);

        assert_eq!(reasoning, "unfinished");
        assert!(content.is_empty());
    }

    #[test]
    fn split_reasoning_prefix_extracts_reasoning_and_trailing_content() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let ReasoningSplit { reasoning, content } = split_reasoning_prefix(
            "<think>deduce</think>answer<tool>tail",
            Some(&markers),
            Some("<tool>"),
            false,
        );

        assert_eq!(reasoning, "deduce");
        assert_eq!(content, "answer");
    }

    #[test]
    fn restore_partial_reasoning_preserves_non_partial_parser_result() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed =
            ParsedChatMessage::new("parsed content".to_owned(), String::new(), Vec::new());

        restore_partial_reasoning(&mut parsed, "<think>unfinished", Some(&markers), false);

        assert_eq!(parsed.content, "parsed content");
        assert!(parsed.reasoning_content.is_empty());
    }

    #[test]
    fn restore_partial_reasoning_preserves_existing_reasoning() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed = ParsedChatMessage::new(
            "parsed content".to_owned(),
            "parsed reasoning".to_owned(),
            Vec::new(),
        );

        restore_partial_reasoning(&mut parsed, "plain response", Some(&markers), true);

        assert_eq!(parsed.content, "parsed content");
        assert_eq!(parsed.reasoning_content, "parsed reasoning");
    }

    #[test]
    fn restore_partial_reasoning_removes_open_marker_from_parser_result() {
        let markers = ReasoningMarkers {
            open: "\n[THINK]\n".to_owned(),
            closes: vec!["[/THINK]".to_owned()],
        };
        let mut parsed = ParsedChatMessage::new(
            String::new(),
            "[THINK]parsed reasoning".to_owned(),
            Vec::new(),
        );

        restore_partial_reasoning(&mut parsed, "complete response", Some(&markers), true);

        assert!(parsed.content.is_empty());
        assert_eq!(parsed.reasoning_content, "parsed reasoning");
    }

    #[test]
    fn restore_partial_reasoning_preserves_unclosed_reasoning_whitespace() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed =
            ParsedChatMessage::new(String::new(), "normalized reasoning".to_owned(), Vec::new());

        restore_partial_reasoning(&mut parsed, "<think>\n\nreasoning", Some(&markers), true);

        assert!(parsed.content.is_empty());
        assert_eq!(parsed.reasoning_content, "\n\nreasoning");
    }

    #[test]
    fn restore_partial_reasoning_preserves_closed_reasoning_whitespace() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed = ParsedChatMessage::new(
            "answer".to_owned(),
            "normalized reasoning".to_owned(),
            Vec::new(),
        );

        restore_partial_reasoning(
            &mut parsed,
            "<think>\n\nreasoning</think>answer",
            Some(&markers),
            true,
        );

        assert_eq!(parsed.content, "answer");
        assert_eq!(parsed.reasoning_content, "\n\nreasoning");
    }

    #[test]
    fn restore_partial_reasoning_removes_open_marker_after_parser_whitespace() {
        let markers = ReasoningMarkers {
            open: "\n[THINK]\n".to_owned(),
            closes: vec!["[/THINK]".to_owned()],
        };
        let mut parsed = ParsedChatMessage::new(
            String::new(),
            "\n[THINK]parsed reasoning".to_owned(),
            Vec::new(),
        );

        restore_partial_reasoning(&mut parsed, "complete response", Some(&markers), true);

        assert!(parsed.content.is_empty());
        assert_eq!(parsed.reasoning_content, "parsed reasoning");
    }

    #[test]
    fn restore_partial_reasoning_preserves_result_without_open_marker() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed =
            ParsedChatMessage::new("parsed content".to_owned(), String::new(), Vec::new());

        restore_partial_reasoning(&mut parsed, "unfinished", Some(&markers), true);

        assert_eq!(parsed.content, "parsed content");
        assert!(parsed.reasoning_content.is_empty());
    }

    #[test]
    fn restore_partial_reasoning_recovers_unclosed_reasoning() {
        let markers = ReasoningMarkers {
            open: "<think>".to_owned(),
            closes: vec!["</think>".to_owned()],
        };
        let mut parsed =
            ParsedChatMessage::new("<think>unfinished".to_owned(), String::new(), Vec::new());

        restore_partial_reasoning(&mut parsed, "<think>unfinished", Some(&markers), true);

        assert!(parsed.content.is_empty());
        assert_eq!(parsed.reasoning_content, "unfinished");
    }

    #[test]
    fn outcome_from_via_ffi_result_recognized_synthesizes_tool_call_ids() {
        let parsed = ParsedChatMessage::new(
            "answer".to_owned(),
            String::new(),
            vec![ParsedToolCall::new(
                String::new(),
                "tool".to_owned(),
                ToolCallArguments::default(),
            )],
        );

        let outcome = outcome_from_via_ffi_result(Ok(parsed), "answer", false);

        assert_eq!(
            outcome.unwrap(),
            ChatMessageParseOutcome::Recognized(ParsedChatMessage::new(
                "answer".to_owned(),
                String::new(),
                vec![ParsedToolCall::new(
                    "call_0".to_owned(),
                    "tool".to_owned(),
                    ToolCallArguments::default(),
                )],
            ))
        );
    }

    #[test]
    fn outcome_from_via_ffi_result_message_unrecognized_is_unrecognized_with_raw_message() {
        let outcome = outcome_from_via_ffi_result(
            Err(ParseChatMessageError::MessageUnrecognized {
                message: "boom".to_owned(),
            }),
            "garbled",
            true,
        );

        assert_eq!(
            outcome.unwrap(),
            ChatMessageParseOutcome::Unrecognized(RawChatMessage {
                text: "garbled".to_owned(),
                is_partial: true,
                ffi_error_message: "boom".to_owned(),
            })
        );
    }

    #[test]
    fn outcome_from_via_ffi_result_parser_creation_failure_propagates() {
        let outcome = outcome_from_via_ffi_result(
            Err(ParseChatMessageError::ParserCreationFailed {
                message: "the parser could not be built".to_owned(),
            }),
            "garbled",
            true,
        );

        assert_eq!(
            discriminant(&outcome.unwrap_err()),
            discriminant(&ParseChatMessageError::ParserCreationFailed {
                message: String::new()
            })
        );
    }

    #[test]
    fn outcome_from_via_ffi_result_other_error_propagates() {
        let outcome = outcome_from_via_ffi_result(Err(ParseChatMessageError::NoVocab), "x", false);

        assert_eq!(
            discriminant(&outcome.unwrap_err()),
            discriminant(&ParseChatMessageError::NoVocab)
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

        let Err(ParseChatMessageError::NotEnoughMemory) = result else {
            panic!("an error-string allocation failure must map to NotEnoughMemory");
        };
    }

    #[test]
    fn parsed_chat_free_llama_cpp_out_of_memory_is_preserved() {
        let result = unsafe {
            parsed_chat_free_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_FREE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
            )
        };

        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = result else {
            panic!("a llama.cpp allocation failure must be reported as its own variant");
        };
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

        let Err(ParseChatMessageError::DestructorFailed { message }) = result else {
            panic!("a throwing destructor must surface its message");
        };

        assert_eq!(message, "the destructor threw");
    }

    #[test]
    fn parsed_chat_free_unknown_status_is_preserved() {
        let result = unsafe { parsed_chat_free_status_to_result(255, ptr::null_mut()) };

        let Err(ParseChatMessageError::FfiStatus(status_error)) = result else {
            panic!("an unrecognized status must be preserved verbatim");
        };

        assert_eq!(
            status_error,
            crate::FfiStatusError {
                operation: "llama_rs_parsed_chat_free",
                code: 255,
            }
        );
    }

    #[test]
    fn parse_chat_message_status_to_result_maps_every_contract_status() {
        let mut out_error_slot: *mut c_char = ptr::null_mut();
        let outcome_0 = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_TOOLS_PARSER_ARG,
                ptr::null_mut(),
                &raw mut out_error_slot,
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_TOOLS_PARSER_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null tools parser argument",
            }
        );
        let outcome_1 = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG,
                ptr::null_mut(),
                &raw mut out_error_slot,
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!("LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_INPUT_ARG must map to a contract error");
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null input argument",
            }
        );
        let outcome_2 = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG,
                ptr::null_mut(),
                &raw mut out_error_slot,
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_2)) = outcome_2 else {
            panic!("LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_HANDLE_ARG must map to a contract error");
        };
        assert_eq!(
            contract_2,
            crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null out_handle argument",
            }
        );
        let outcome_3 = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG,
                ptr::null_mut(),
                &raw mut out_error_slot,
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_3)) = outcome_3 else {
            panic!("LLAMA_RS_PARSE_CHAT_MESSAGE_NULL_OUT_ERROR_ARG must map to a contract error");
        };
        assert_eq!(
            contract_3,
            crate::FfiContractError {
                operation: "llama_rs_parse_chat_message",
                detail: "was given a null out_error argument",
            }
        );
        let outcome_4 = unsafe {
            parse_chat_message_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                &raw mut out_error_slot,
            )
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_4 else {
            panic!(
                "LLAMA_RS_PARSE_CHAT_MESSAGE_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_content_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!("LLAMA_RS_PARSED_CHAT_CONTENT_NULL_HANDLE_ARG must map to a contract error");
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_content",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!("LLAMA_RS_PARSED_CHAT_CONTENT_NULL_OUT_STRING_ARG must map to a contract error");
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_content",
                detail: "was given a null out_string argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_CONTENT_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_CONTENT_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_reasoning_content_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_HANDLE_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_reasoning_content_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_NULL_OUT_STRING_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_reasoning_content",
                detail: "was given a null out_string argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_reasoning_content_status_to_result(llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_LLAMA_CPP_OUT_OF_MEMORY, ptr::null_mut(), ptr::null_mut())
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_REASONING_CONTENT_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_tool_call_count_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG,
                0,
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_HANDLE_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG,
                0,
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_NULL_OUT_COUNT_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_count",
                detail: "was given a null out_count argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_tool_call_count_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_COUNT_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_tool_call_id_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_HANDLE_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_NULL_OUT_STRING_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_id",
                detail: "was given a null out_string argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_tool_call_id_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ID_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_tool_call_name_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_HANDLE_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_NULL_OUT_STRING_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_name",
                detail: "was given a null out_string argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_tool_call_name_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_NAME_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    #[test]
    fn parsed_chat_tool_call_arguments_status_to_result_maps_every_contract_status() {
        let outcome_0 = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let Err(ParseChatMessageError::FfiContract(contract_0)) = outcome_0 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_HANDLE_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_0,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                detail: "was given a null handle argument",
            }
        );
        let outcome_1 = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG, 0, ptr::null_mut(), ptr::null_mut())
        };
        let Err(ParseChatMessageError::FfiContract(contract_1)) = outcome_1 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NULL_OUT_STRING_ARG must map to a contract error"
            );
        };
        assert_eq!(
            contract_1,
            crate::FfiContractError {
                operation: "llama_rs_parsed_chat_tool_call_arguments",
                detail: "was given a null out_string argument",
            }
        );
        let outcome_2 = unsafe {
            parsed_chat_tool_call_arguments_status_to_result(llama_cpp_bindings_sys::LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_LLAMA_CPP_OUT_OF_MEMORY, 0, ptr::null_mut(), ptr::null_mut())
        };
        let Err(ParseChatMessageError::LlamaCppOutOfMemory) = outcome_2 else {
            panic!(
                "LLAMA_RS_PARSED_CHAT_TOOL_CALL_ARGUMENTS_LLAMA_CPP_OUT_OF_MEMORY must map to LlamaCppOutOfMemory"
            );
        };
    }

    fn tools_parser_creation_error(
        status: llama_cpp_bindings_sys::llama_rs_chat_tools_parser_create_status,
        message: Option<&CStr>,
    ) -> ParseChatMessageError {
        let mut out_error = message.map_or(ptr::null_mut(), |message| unsafe {
            llama_cpp_bindings_sys::llama_rs_string_dup(message.as_ptr())
        });

        let error = unsafe {
            chat_tools_parser_create_status_to_result(status, ptr::null_mut(), &raw mut out_error)
        }
        .expect_err("the status must map to an error");

        assert!(out_error.is_null(), "a reported message must be reclaimed");

        error
    }

    #[test]
    fn tools_parser_creation_reports_a_success_without_a_parser_as_a_contract_error() {
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_OK,
                None
            ),
            ParseChatMessageError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "success status contained a null tools parser handle",
            })
        );
    }

    #[test]
    fn tools_parser_creation_reports_tools_that_are_not_an_array() {
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_TOOLS_NOT_AN_ARRAY,
                None
            ),
            ParseChatMessageError::ToolsNotAnArray
        );
    }

    #[test]
    fn tools_parser_creation_reports_running_out_of_memory() {
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_ERROR_STRING_ALLOCATION_FAILED,
                None
            ),
            ParseChatMessageError::NotEnoughMemory
        );
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_LLAMA_CPP_OUT_OF_MEMORY,
                None
            ),
            ParseChatMessageError::LlamaCppOutOfMemory
        );
    }

    #[test]
    fn tools_parser_creation_reports_the_build_failure_message() {
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                Some(c"tools do not fit the template")
            ),
            ParseChatMessageError::ToolsParserBuildFailed {
                message: "tools do not fit the template".to_owned()
            }
        );
    }

    #[test]
    fn tools_parser_creation_reports_a_build_failure_without_a_message_as_a_contract_error() {
        assert_eq!(
            tools_parser_creation_error(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                None
            ),
            ParseChatMessageError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_chat_tools_parser_create",
                detail: "reported a thrown C++ exception without an error message",
            })
        );
    }

    #[test]
    fn tools_parser_creation_reports_every_null_argument_as_a_contract_error() {
        for (status, detail) in [
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_PARSER_ARG,
                "was given a null parser argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_TOOLS_JSON_ARG,
                "was given a null tools_json argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_OUT_TOOLS_PARSER_ARG,
                "was given a null out_tools_parser argument",
            ),
            (
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_CREATE_NULL_OUT_ERROR_ARG,
                "was given a null out_error argument",
            ),
        ] {
            assert_eq!(
                tools_parser_creation_error(status, None),
                ParseChatMessageError::FfiContract(crate::FfiContractError {
                    operation: "llama_rs_chat_tools_parser_create",
                    detail,
                })
            );
        }
    }

    #[test]
    fn tools_parser_creation_preserves_an_unknown_status() {
        assert_eq!(
            tools_parser_creation_error(255, None),
            ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_chat_tools_parser_create",
                code: 255,
            })
        );
    }

    #[test]
    fn tools_parser_release_maps_every_status() {
        assert_eq!(
            unsafe {
                chat_tools_parser_free_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_OK,
                    ptr::null_mut(),
                )
            },
            Ok(())
        );
        assert_eq!(
            unsafe {
                chat_tools_parser_free_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_ERROR_STRING_ALLOCATION_FAILED,
                    ptr::null_mut(),
                )
            },
            Err(ParseChatMessageError::NotEnoughMemory)
        );
        assert_eq!(
            unsafe {
                chat_tools_parser_free_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_LLAMA_CPP_OUT_OF_MEMORY,
                    ptr::null_mut(),
                )
            },
            Err(ParseChatMessageError::LlamaCppOutOfMemory)
        );
        assert_eq!(
            unsafe {
                chat_tools_parser_free_status_to_result(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_TOOLS_PARSER_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
                    llama_cpp_bindings_sys::llama_rs_string_dup(c"destructor failed".as_ptr()),
                )
            },
            Err(ParseChatMessageError::DestructorFailed {
                message: "destructor failed".to_owned()
            })
        );
        assert_eq!(
            unsafe { chat_tools_parser_free_status_to_result(255, ptr::null_mut()) },
            Err(ParseChatMessageError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_chat_tools_parser_free",
                code: 255,
            }))
        );
    }
}
