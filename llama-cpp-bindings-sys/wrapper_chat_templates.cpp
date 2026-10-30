#include "wrapper_chat_templates.h"
#include "wrapper_chat_templates_handle.h"
#include "wrapper_report_current_exception.h"

#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/jinja/caps.h"
#include "llama.cpp/common/jinja/lexer.h"
#include "llama.cpp/common/jinja/parser.h"
#include "llama.cpp/common/jinja/runtime.h"
#include "llama.cpp/include/llama.h"

#include <memory>
#include <string>

namespace {
auto template_supports_tool_calls(const std::string & source) -> bool {
    jinja::lexer lexer;
    jinja::program program = jinja::parse_from_tokens(lexer.tokenize(source));

    return jinja::caps_get(program).supports_tool_calls;
}
}  // namespace

extern "C" auto llama_rs_chat_templates_init(
    const struct llama_model * model,
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
    if (out_templates == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_TEMPLATES_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_NULL_OUT_ERROR_ARG;
    }
    if (llama_model_chat_template(model, nullptr) == nullptr) {
        return LLAMA_RS_CHAT_TEMPLATES_INIT_MODEL_HAS_NO_CHAT_TEMPLATE;
    }

    try {
        auto templates = std::make_unique<llama_rs_chat_templates>();
        templates->templates = common_chat_templates_init(model, std::string());

        const std::string tool_use_source = common_chat_templates_source(templates->templates.get(), "tool_use");
        templates->supports_tool_calls = template_supports_tool_calls(
            tool_use_source.empty() ? common_chat_templates_source(templates->templates.get()) : tool_use_source);

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
    const std::unique_ptr<llama_rs_chat_templates> reclaimed(templates);
}

extern "C" auto llama_rs_chat_templates_supports_tool_calls(llama_rs_chat_templates_handle templates) -> bool {
    return templates->supports_tool_calls;
}
