#pragma once

#include <llama.h>
#include "wrapper_chat_templates.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t llama_rs_model_declared_pooling_type(const struct llama_model * model);

bool llama_rs_model_outputs_text(const struct llama_model * model, bool projector_chats, bool projector_speaks);

typedef struct llama_rs_chat_templates_capabilities {
    bool supports_parallel_tool_calls;
    bool supports_system_role;
    bool supports_thinking;
    bool supports_tool_calls;
    bool supports_tools;
} llama_rs_chat_templates_capabilities;

typedef enum llama_rs_chat_templates_capabilities_status {
    LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_OK = 0,
    LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_chat_templates_capabilities_status;

llama_rs_chat_templates_capabilities_status llama_rs_chat_templates_capabilities_of(
    llama_rs_chat_templates_handle templates,
    llama_rs_chat_templates_capabilities * out_capabilities,
    char ** out_error);

typedef struct llama_rs_sampling_defaults {
    float min_p;
    float penalty_frequency;
    int32_t penalty_last_n;
    float penalty_presence;
    float penalty_repeat;
    float temperature;
    int32_t top_k;
    float top_p;
} llama_rs_sampling_defaults;

llama_rs_sampling_defaults llama_rs_model_sampling_defaults(const struct llama_model * model);

#ifdef __cplusplus
}
#endif
