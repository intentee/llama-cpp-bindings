#include "wrapper_chat_templates.h"
#include "wrapper_chat_templates_handle.h"
#include "wrapper_report_current_exception.h"

#include <common/chat.h>
#include <llama.h>

#include <memory>
#include <string>

extern "C" auto llama_rs_chat_templates_init(
    const struct llama_model * model,
    const char * template_override,
    llama_rs_chat_templates_handle * out_templates,
    char ** out_error) -> llama_rs_chat_templates_init_status {
    if (out_templates != nullptr) {
        *out_templates = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_MODEL_ARG;
    }
    if (template_override == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_TEMPLATE_OVERRIDE_ARG;
    }
    if (out_templates == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_TEMPLATES_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_ERROR_ARG;
    }
    if (*template_override == '\0' && llama_model_chat_template(model, nullptr) == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE;
    }

    try {
        auto templates = std::make_unique<llama_rs_chat_templates>();
        templates->templates = common_chat_templates_init(model, std::string(template_override));

        *out_templates = templates.release();

        return LLAMA_RS_CHAT_TEMPLATES_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_TEMPLATES_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_TEMPLATES_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_templates_free(llama_rs_chat_templates_handle templates) -> void {
    std::unique_ptr<llama_rs_chat_templates>(templates).reset();
}
