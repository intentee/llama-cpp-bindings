#include "wrapper_fit.h"
#include <llama.h>
#include <ggml.h>
#include "wrapper_report_current_exception.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <common/common.h>
#include <common/fit.h>

extern "C" auto llama_rs_fit_defaults() -> llama_rs_fit_defaults_values {
    try {
        const common_params llama_cpp_defaults{};

        return llama_rs_fit_defaults_values{
            llama_cpp_defaults.fit_params_target.front(),
            static_cast<uint32_t>(llama_cpp_defaults.fit_params_min_ctx),
        };
    } catch (...) {
        std::abort();
    }
}

extern "C" auto llama_rs_fit_params(
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
    char ** out_error) -> llama_rs_fit_params_status {
    *out_error = nullptr;
    *out_unrecognized_status_code = 0;

    try {
        common_fit_extra_model llama_cpp_extra{};
        if (extra != nullptr) {
            llama_cpp_extra.paths_model  = {extra->path_model};
            llama_cpp_extra.mparams      = extra->mparams;
            llama_cpp_extra.cparams      = extra->cparams;
            llama_cpp_extra.shares_model = extra->shares_model;
        }

        const common_params_fit_status status = common_fit_params(
            {path_model}, mparams, cparams, tensor_split, tensor_buft_overrides,
            margins, n_ctx_min, extra == nullptr ? nullptr : &llama_cpp_extra, log_level);
        switch (status) {
            case COMMON_PARAMS_FIT_STATUS_SUCCESS:
                return LLAMA_RS_FIT_PARAMS_OK;
            case COMMON_PARAMS_FIT_STATUS_FAILURE:
                return LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_FAILURE;
            case COMMON_PARAMS_FIT_STATUS_ERROR:
                return LLAMA_RS_FIT_PARAMS_LLAMA_CPP_REPORTED_ERROR;
        }
        *out_unrecognized_status_code = static_cast<int32_t>(status);
        return LLAMA_RS_FIT_PARAMS_LLAMA_CPP_RETURNED_UNRECOGNIZED_STATUS_CODE;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_FIT_PARAMS_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_FIT_PARAMS_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_FIT_PARAMS_ERROR_STRING_ALLOCATION_FAILED);
    }
}
