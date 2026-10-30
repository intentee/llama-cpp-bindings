#include "wrapper_parsed_chat.h"
#include "wrapper_parsed_chat_handle.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <memory>

extern "C" auto llama_rs_parsed_chat_free(llama_rs_parsed_chat_handle handle) -> void {
    std::unique_ptr<llama_rs_parsed_chat>(handle).reset();
}

extern "C" auto llama_rs_parsed_chat_content(llama_rs_parsed_chat_handle handle, size_t * out_length)
    -> const char * {
    return llama_rs_string_view(handle->message.content, out_length);
}

extern "C" auto llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(handle->message.reasoning_content, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_count(llama_rs_parsed_chat_handle handle) -> size_t {
    return handle->message.tool_calls.size();
}

extern "C" auto llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(handle->message.tool_calls[index].id, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(handle->message.tool_calls[index].name, out_length);
}

extern "C" auto llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(handle->message.tool_calls[index].arguments, out_length);
}
