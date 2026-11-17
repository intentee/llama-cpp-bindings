#include "wrapper_decision.h"
#include "wrapper_report_current_exception.h"
#include "wrapper_utils.h"

#include <cstdint>
#include <stdexcept>

#include <common/common.h>
#include <common/jinja/lexer.h>
#include <common/jinja/parser.h>
#include <common/jinja/runtime.h>
#include <common/jinja/value.h>
#include <common/json.h>
#include <llama.h>
#include <src/llama-ext.h>

namespace {

static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_NONE) == static_cast<int>(COMMON_DECISION_TYPE_NONE));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_OPENJEV) == static_cast<int>(COMMON_DECISION_TYPE_OPENJEV));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LEV) == static_cast<int>(COMMON_DECISION_TYPE_LEV));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_KEV) == static_cast<int>(COMMON_DECISION_TYPE_KEV));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_NIMBLE) == static_cast<int>(COMMON_DECISION_TYPE_NIMBLE));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LAYA) == static_cast<int>(COMMON_DECISION_TYPE_LAYA));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_CLEF) == static_cast<int>(COMMON_DECISION_TYPE_CLEF));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_PPLX_DECIDER) == static_cast<int>(COMMON_DECISION_TYPE_PPLX_DECIDER));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_VLLM_DECISION2) == static_cast<int>(COMMON_DECISION_TYPE_VLLM_DECISION2));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_VLLM_D3) == static_cast<int>(COMMON_DECISION_TYPE_VLLM_D3));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LFM2_D1) == static_cast<int>(COMMON_DECISION_TYPE_LFM2_D1));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LFM2_D1_OMNI) == static_cast<int>(COMMON_DECISION_TYPE_LFM2_D1_OMNI));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_GRANITE_SWITCH) == static_cast<int>(COMMON_DECISION_TYPE_GRANITE_SWITCH));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_H2O_LIGHTNING) == static_cast<int>(COMMON_DECISION_TYPE_H2O_LIGHTNING));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_MINICPM_OCR_JEV) == static_cast<int>(COMMON_DECISION_TYPE_MINICPM_OCR_JEV));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LFM3_D1) == static_cast<int>(COMMON_DECISION_TYPE_LFM3_D1));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_LFM3_D1_OMNI) == static_cast<int>(COMMON_DECISION_TYPE_LFM3_D1_OMNI));
static_assert(static_cast<int>(LLAMA_RS_DECISION_TYPE_UNKNOWN) == static_cast<int>(COMMON_DECISION_TYPE_UNKNOWN));

auto mirrored_decision_type(const common_decision_type decision_type) -> llama_rs_decision_type {
    return static_cast<llama_rs_decision_type>(decision_type);
}

}  // namespace

extern "C" auto llama_rs_model_decision_type(
    const struct llama_model * model,
    llama_rs_decision_type * out_decision_type,
    char ** out_error) -> llama_rs_model_decision_type_status {
    if (out_decision_type != nullptr) {
        *out_decision_type = LLAMA_RS_DECISION_TYPE_NONE;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_MODEL_DECISION_TYPE_NULL_MODEL_ARG;
    }
    if (out_decision_type == nullptr) {
        return LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_DECISION_TYPE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_MODEL_DECISION_TYPE_NULL_OUT_ERROR_ARG;
    }

    try {
        *out_decision_type = mirrored_decision_type(common_get_decision_type(model));

        return LLAMA_RS_MODEL_DECISION_TYPE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MODEL_DECISION_TYPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MODEL_DECISION_TYPE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_model_decision_label(const struct llama_model * model, uint32_t index) -> const char * {
    return llama_model_decision_label(model, index);
}

extern "C" auto llama_rs_model_classifier_free_guidance(const struct llama_model * model) -> bool {
    return llama_model_classifier_free_guidance(model);
}

extern "C" auto llama_rs_read_gguf_info(
    const char * path,
    llama_rs_gguf_info * out_info,
    char ** out_error) -> llama_rs_read_gguf_info_status {
    if (out_info != nullptr) {
        *out_info = llama_rs_gguf_info{LLAMA_RS_DECISION_TYPE_UNKNOWN, 0, false};
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (path == nullptr) {
        return LLAMA_RS_READ_GGUF_INFO_NULL_PATH_ARG;
    }
    if (out_info == nullptr) {
        return LLAMA_RS_READ_GGUF_INFO_NULL_OUT_INFO_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_READ_GGUF_INFO_NULL_OUT_ERROR_ARG;
    }

    try {
        const common_gguf_info info = common_get_gguf_info(path);

        *out_info = llama_rs_gguf_info{
            mirrored_decision_type(info.decision_type),
            info.n_ctx_train,
            info.has_chat_tmpl,
        };

        return LLAMA_RS_READ_GGUF_INFO_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_READ_GGUF_INFO_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_READ_GGUF_INFO_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_render_template(
    const char * template_source,
    const char * globals_json,
    char ** out_rendered,
    char ** out_error) -> llama_rs_render_template_status {
    if (out_rendered != nullptr) {
        *out_rendered = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (template_source == nullptr) {
        return LLAMA_RS_RENDER_TEMPLATE_NULL_TEMPLATE_SOURCE_ARG;
    }
    if (globals_json == nullptr) {
        return LLAMA_RS_RENDER_TEMPLATE_NULL_GLOBALS_JSON_ARG;
    }
    if (out_rendered == nullptr) {
        return LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_RENDERED_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_RENDER_TEMPLATE_NULL_OUT_ERROR_ARG;
    }

    try {
        const common_json globals = common_json::parse(globals_json);

        if (!globals.is_object()) {
            return LLAMA_RS_RENDER_TEMPLATE_GLOBALS_NOT_A_JSON_OBJECT;
        }

        jinja::lexer_result tokenized;
        jinja::program program;

        try {
            jinja::lexer lexer;
            tokenized = lexer.tokenize(template_source);
            program = jinja::parse_from_tokens(tokenized);
        } catch (const std::runtime_error & err) {
            return llama_rs_report_exception_message(
                out_error,
                err.what(),
                LLAMA_RS_RENDER_TEMPLATE_INVALID_TEMPLATE,
                LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED);
        }

        jinja::context context(tokenized.source);
        jinja::global_from_json(context, globals, false);
        jinja::runtime runtime(context);
        const jinja::value results = runtime.execute(program);

        *out_rendered = llama_rs_dup_string(jinja::runtime::gather_string_parts(results)->as_string().str());
        if (*out_rendered == nullptr) {
            return LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_RENDER_TEMPLATE_OK;
    } catch (const common_json_error & err) {
        return llama_rs_report_exception_message(
            out_error,
            err.what(),
            LLAMA_RS_RENDER_TEMPLATE_INVALID_JSON,
            LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED);
    } catch (const jinja::rethrown_exception & err) {
        return llama_rs_report_exception_message(
            out_error,
            err.what(),
            LLAMA_RS_RENDER_TEMPLATE_EXECUTION_FAILED,
            LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED);
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_RENDER_TEMPLATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_RENDER_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED);
    }
}
