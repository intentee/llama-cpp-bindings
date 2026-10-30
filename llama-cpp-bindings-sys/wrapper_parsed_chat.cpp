#include "wrapper_parsed_chat.h"
#include "wrapper_parsed_chat_handle.h"
#include "wrapper_report_current_exception.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <memory>

extern "C" auto llama_rs_parsed_chat_free(
    llama_rs_parsed_chat_handle handle,
    char ** out_error) -> llama_rs_parsed_chat_free_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    try {
        const std::unique_ptr<llama_rs_parsed_chat> reclaimed(handle);
        return LLAMA_RS_PARSED_CHAT_FREE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_PARSED_CHAT_FREE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
            LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED);
    }
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
