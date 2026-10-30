#pragma once

#include "llama.cpp/include/llama.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_parsed_chat;
typedef struct llama_rs_parsed_chat * llama_rs_parsed_chat_handle;

typedef enum llama_rs_parsed_chat_free_status {
    LLAMA_RS_PARSED_CHAT_FREE_OK = 0,
    LLAMA_RS_PARSED_CHAT_FREE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_PARSED_CHAT_FREE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_PARSED_CHAT_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
} llama_rs_parsed_chat_free_status;

llama_rs_parsed_chat_free_status llama_rs_parsed_chat_free(
    llama_rs_parsed_chat_handle handle,
    char ** out_error);

const char * llama_rs_parsed_chat_content(llama_rs_parsed_chat_handle handle, size_t * out_length);

const char * llama_rs_parsed_chat_reasoning_content(
    llama_rs_parsed_chat_handle handle,
    size_t * out_length);

size_t llama_rs_parsed_chat_tool_call_count(llama_rs_parsed_chat_handle handle);

const char * llama_rs_parsed_chat_tool_call_id(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length);

const char * llama_rs_parsed_chat_tool_call_name(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length);

const char * llama_rs_parsed_chat_tool_call_arguments(
    llama_rs_parsed_chat_handle handle,
    size_t index,
    size_t * out_length);

#ifdef __cplusplus
}
#endif
