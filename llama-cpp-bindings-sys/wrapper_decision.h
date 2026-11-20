#pragma once

#include <llama.h>

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_decision_type {
    LLAMA_RS_DECISION_TYPE_NONE = 0,
    LLAMA_RS_DECISION_TYPE_OPENJEV,
    LLAMA_RS_DECISION_TYPE_LEV,
    LLAMA_RS_DECISION_TYPE_KEV,
    LLAMA_RS_DECISION_TYPE_NIMBLE,
    LLAMA_RS_DECISION_TYPE_LAYA,
    LLAMA_RS_DECISION_TYPE_CLEF,
    LLAMA_RS_DECISION_TYPE_PPLX_DECIDER,
    LLAMA_RS_DECISION_TYPE_VLLM_DECISION2,
    LLAMA_RS_DECISION_TYPE_VLLM_D3,
    LLAMA_RS_DECISION_TYPE_LFM2_D1,
    LLAMA_RS_DECISION_TYPE_LFM2_D1_OMNI,
    LLAMA_RS_DECISION_TYPE_GRANITE_SWITCH,
    LLAMA_RS_DECISION_TYPE_H2O_LIGHTNING,
    LLAMA_RS_DECISION_TYPE_MINICPM_OCR_JEV,
    LLAMA_RS_DECISION_TYPE_LFM3_D1,
    LLAMA_RS_DECISION_TYPE_LFM3_D1_OMNI,
    LLAMA_RS_DECISION_TYPE_UNKNOWN,
} llama_rs_decision_type;

typedef enum llama_rs_model_decision_type_status {
    LLAMA_RS_MODEL_DECISION_TYPE_OK = 0,
    LLAMA_RS_MODEL_DECISION_TYPE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_model_decision_type_status;

llama_rs_model_decision_type_status llama_rs_model_decision_type(
    const struct llama_model * model,
    llama_rs_decision_type * out_decision_type,
    char ** out_error);

const char * llama_rs_model_decision_label(const struct llama_model * model, uint32_t index);

bool llama_rs_model_classifier_free_guidance(const struct llama_model * model);

typedef struct llama_rs_gguf_info {
    llama_rs_decision_type decision_type;
    uint32_t n_ctx_train;
    bool has_chat_template;
} llama_rs_gguf_info;

typedef enum llama_rs_read_gguf_info_status {
    LLAMA_RS_READ_GGUF_INFO_OK = 0,
    LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_read_gguf_info_status;

llama_rs_read_gguf_info_status llama_rs_read_gguf_info(
    const char * path,
    llama_rs_gguf_info * out_info,
    char ** out_error);

typedef enum llama_rs_render_template_status {
    LLAMA_RS_RENDER_TEMPLATE_OK = 0,
    LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_RENDER_TEMPLATE_INVALID_JSON,
    LLAMA_RS_RENDER_TEMPLATE_GLOBALS_NOT_A_JSON_OBJECT,
    LLAMA_RS_RENDER_TEMPLATE_INVALID_TEMPLATE,
    LLAMA_RS_RENDER_TEMPLATE_EXECUTION_FAILED,
    LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_render_template_status;

llama_rs_render_template_status llama_rs_render_template(
    const char * template_source,
    const char * globals_json,
    char ** out_rendered,
    char ** out_error);

#ifdef __cplusplus
}
#endif
