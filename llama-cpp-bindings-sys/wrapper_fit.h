#pragma once

#include <ggml.h>
#include <llama.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_fit_params_status {
    LLAMA_RS_FIT_PARAMS_OK = 0,
    LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE,
    LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR,
    LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE,
    LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_fit_params_status;

typedef enum llama_rs_fit_model_traits_status {
    LLAMA_RS_FIT_MODEL_TRAITS_READ = 0,
    LLAMA_RS_FIT_MODEL_TRAITS_MODEL_UNLOADABLE,
} llama_rs_fit_model_traits_status;

typedef struct llama_rs_fit_model_traits {
    bool causal_attention;
    bool has_encoder;
    uint32_t n_ctx_train;
} llama_rs_fit_model_traits;

typedef struct llama_rs_fit_defaults_values {
    size_t margin_per_device;
    uint32_t n_ctx_min;
} llama_rs_fit_defaults_values;

typedef struct llama_rs_fit_extra_model {
    const char * path_model;
    struct llama_model_params * mparams;
    struct llama_context_params * cparams;
    bool shares_model;
} llama_rs_fit_extra_model;

llama_rs_fit_defaults_values llama_rs_fit_defaults(void);

llama_rs_fit_model_traits_status llama_rs_fit_model_traits_of(
    const char * path_model,
    struct llama_model_params mparams,
    llama_rs_fit_model_traits * out_traits);

llama_rs_fit_params_status llama_rs_fit_params(
    const char * path_model,
    struct llama_model_params * mparams,
    struct llama_context_params * cparams,
    float * tensor_split,
    struct llama_model_tensor_buft_override * tensor_buft_overrides,
    size_t * margins,
    uint32_t n_ctx_min,
    const llama_rs_fit_extra_model * extra,
    enum ggml_log_level log_level,
    int32_t * out_unrecognized_status_code,
    char ** out_error);

#ifdef __cplusplus
}
#endif
