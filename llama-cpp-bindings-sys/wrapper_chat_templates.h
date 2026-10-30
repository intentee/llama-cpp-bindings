#pragma once

#include "llama.cpp/include/llama.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_chat_templates;
typedef struct llama_rs_chat_templates * llama_rs_chat_templates_handle;

typedef enum llama_rs_chat_templates_init_status {
    LLAMA_RS_CHAT_TEMPLATES_INIT_OK = 0,
    LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_MODEL_ARG,
    LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_TEMPLATES_ARG,
    LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_ERROR_ARG,
    LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE,
    LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_chat_templates_init_status;

llama_rs_chat_templates_init_status llama_rs_chat_templates_init(
    const struct llama_model * model,
    llama_rs_chat_templates_handle * out_templates,
    char ** out_error);

typedef enum llama_rs_chat_templates_free_status {
    LLAMA_RS_CHAT_TEMPLATES_FREE_OK = 0,
    LLAMA_RS_CHAT_TEMPLATES_FREE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_TEMPLATES_FREE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_TEMPLATES_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
} llama_rs_chat_templates_free_status;

llama_rs_chat_templates_free_status llama_rs_chat_templates_free(
    llama_rs_chat_templates_handle templates,
    char ** out_error);

bool llama_rs_chat_templates_supports_tool_calls(llama_rs_chat_templates_handle templates);

#ifdef __cplusplus
}
#endif
