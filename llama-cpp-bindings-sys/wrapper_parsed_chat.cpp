#include "wrapper_parsed_chat.h"
#include "wrapper_fault_injection.h"
#include "wrapper_parsed_chat_handle.h"

#include <cstddef>
#include <memory>

extern "C" auto llama_rs_parsed_chat_free(llama_rs_parsed_chat_handle handle) -> void {
    std::unique_ptr<llama_rs_parsed_chat>(handle).reset();
}

extern "C" auto llama_rs_parsed_chat_content(llama_rs_parsed_chat_handle handle, size_t * out_length)
    -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_PARSED_CHAT_CONTENT_NOT_UTF8, handle->message.content, out_length);
}

extern "C" auto llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_PARSED_CHAT_REASONING_CONTENT_NOT_UTF8, handle->message.reasoning_content, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_count(llama_rs_parsed_chat_handle handle) -> size_t {
    return handle->message.tool_calls.size();
}

extern "C" auto llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_ID_NOT_UTF8, handle->message.tool_calls[index].id, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_NAME_NOT_UTF8, handle->message.tool_calls[index].name, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NOT_UTF8, handle->message.tool_calls[index].arguments, out_length);
}
