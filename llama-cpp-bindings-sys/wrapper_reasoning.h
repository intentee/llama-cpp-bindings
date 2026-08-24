#pragma once

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_detect_reasoning_markers_status {
    LLAMA_RS_DETECT_REASONING_MARKERS_OK = 0,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_MODEL_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_MARKERS_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_ERROR_ARG,
    LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_detect_reasoning_markers_status;

typedef struct llama_rs_reasoning_markers llama_rs_reasoning_markers;

llama_rs_detect_reasoning_markers_status llama_rs_detect_reasoning_markers(
    const struct llama_model * model,
    llama_rs_reasoning_markers ** out_markers,
    char ** out_error);

const char * llama_rs_reasoning_markers_open(const llama_rs_reasoning_markers * markers);
size_t llama_rs_reasoning_markers_close_count(const llama_rs_reasoning_markers * markers);
const char * llama_rs_reasoning_markers_close_at(
    const llama_rs_reasoning_markers * markers,
    size_t index);
typedef enum llama_rs_reasoning_markers_free_status {
    LLAMA_RS_REASONING_MARKERS_FREE_OK = 0,
    LLAMA_RS_REASONING_MARKERS_FREE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_REASONING_MARKERS_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION,
} llama_rs_reasoning_markers_free_status;

llama_rs_reasoning_markers_free_status llama_rs_reasoning_markers_free(
    llama_rs_reasoning_markers * markers,
    char ** out_error);

typedef enum llama_rs_render_chat_template_status {
    LLAMA_RS_RENDER_CHAT_TEMPLATE_OK = 0,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MODEL_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MESSAGES_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_RENDERED_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_CHAT_TEMPLATE,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_RENDER_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_render_chat_template_status;

llama_rs_render_chat_template_status llama_rs_render_chat_template(
    const struct llama_model * model,
    const char * messages_json,
    int add_generation_prompt,
    int enable_thinking,
    char ** out_rendered,
    char ** out_error);

#ifdef __cplusplus
}
#endif
