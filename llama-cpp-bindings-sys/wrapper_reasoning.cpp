#include "wrapper_reasoning.h"

#include "llama.cpp/common/chat-auto-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/include/llama.h"
#include <nlohmann/json.hpp> // IWYU pragma: keep
#include <nlohmann/json_fwd.hpp>
#include "wrapper_utils.h"

#include <cstddef>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

struct llama_rs_reasoning_markers {
    std::string open;
    std::vector<std::string> closes;
};

namespace {

auto token_text_or_empty(const llama_vocab * vocab, llama_token token) -> std::string {
    if (token == LLAMA_TOKEN_NULL) {
        return {};
    }

    const char * text = llama_vocab_get_text(vocab, token);
    if (text == nullptr) {
        return {};
    }

    return {text};
}

auto find_reasoning_markers(
    const common_chat_template & tmpl,
    const char * tmpl_src,
    std::string * out_start,
    std::vector<std::string> * out_ends) -> bool {
    autoparser::generation_params probe_params;
    probe_params.add_generation_prompt = true;
    probe_params.enable_thinking = true;
    probe_params.is_inference = false;
    probe_params.add_inference = false;
    probe_params.mark_input = false;
    probe_params.messages = nlohmann::ordered_json::array({
        nlohmann::ordered_json{ { "role", "user" }, { "content", "ping" } },
    });

    const std::string tmpl_src_str = tmpl_src;
    if (auto specialized = common_chat_try_specialized_template(tmpl, tmpl_src_str, probe_params)) {
        if (specialized->supports_thinking
            && !specialized->thinking_start_tag.empty()
            && !specialized->thinking_end_tags.empty()) {
            *out_start = std::move(specialized->thinking_start_tag);
            *out_ends = std::move(specialized->thinking_end_tags);
            return true;
        }
    }

    autoparser::autoparser parser;
    parser.analyze_template(tmpl);
    if (parser.reasoning.mode != autoparser::reasoning_mode::NONE
        && !parser.reasoning.start.empty()
        && !parser.reasoning.end.empty()) {
        *out_start = std::move(parser.reasoning.start);
        out_ends->push_back(std::move(parser.reasoning.end));
        return true;
    }

    return false;
}

}  // namespace

extern "C" auto llama_rs_detect_reasoning_markers(
    const struct llama_model * model,
    llama_rs_reasoning_markers ** out_markers,
    char ** out_error) -> llama_rs_detect_reasoning_markers_status {
    if (out_markers != nullptr) {
        *out_markers = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_DETECT_REASONING_MARKERS_NULL_MODEL_ARG;
    }
    if (out_markers == nullptr) {
        return LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_MARKERS_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_DETECT_REASONING_MARKERS_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr) {
            return LLAMA_RS_DETECT_REASONING_MARKERS_OK;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_DETECT_REASONING_MARKERS_OK;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(tmpl_src, bos_token, eos_token);

        auto detected = std::make_unique<llama_rs_reasoning_markers>();
        if (!find_reasoning_markers(tmpl, tmpl_src, &detected->open, &detected->closes)) {
            return LLAMA_RS_DETECT_REASONING_MARKERS_OK;
        }

        *out_markers = detected.release();

        return LLAMA_RS_DETECT_REASONING_MARKERS_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (*out_error == nullptr) {
            return LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (*out_error == nullptr) {
            return LLAMA_RS_DETECT_REASONING_MARKERS_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_DETECT_REASONING_MARKERS_VENDORED_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_reasoning_markers_open(
    const llama_rs_reasoning_markers * markers) -> const char * {
    if (markers == nullptr) {
        return nullptr;
    }
    return markers->open.c_str();
}

extern "C" auto llama_rs_reasoning_markers_close_count(
    const llama_rs_reasoning_markers * markers) -> size_t {
    if (markers == nullptr) {
        return 0;
    }
    return markers->closes.size();
}

extern "C" auto llama_rs_reasoning_markers_close_at(
    const llama_rs_reasoning_markers * markers,
    size_t index) -> const char * {
    if (markers == nullptr || index >= markers->closes.size()) {
        return nullptr;
    }
    return markers->closes[index].c_str();
}

extern "C" auto llama_rs_reasoning_markers_free(
    llama_rs_reasoning_markers * markers,
    char ** out_error) -> llama_rs_reasoning_markers_free_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    try {
        const std::unique_ptr<llama_rs_reasoning_markers> reclaimed(markers);
        return LLAMA_RS_REASONING_MARKERS_FREE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_REASONING_MARKERS_FREE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string(err.what());
            if (*out_error == nullptr) {
                return LLAMA_RS_REASONING_MARKERS_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_REASONING_MARKERS_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    } catch (...) {
        if (out_error != nullptr) {
            *out_error = llama_rs_dup_string("unknown c++ exception");
            if (*out_error == nullptr) {
                return LLAMA_RS_REASONING_MARKERS_FREE_ERROR_STRING_ALLOCATION_FAILED;
            }
        }
        return LLAMA_RS_REASONING_MARKERS_FREE_DESTRUCTOR_THREW_CXX_EXCEPTION;
    }
}

extern "C" auto llama_rs_render_chat_template(
    const struct llama_model * model,
    const char * messages_json,
    int add_generation_prompt,
    int enable_thinking,
    char ** out_rendered,
    char ** out_error) -> llama_rs_render_chat_template_status {
    if (out_rendered != nullptr) {
        *out_rendered = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (model == nullptr) {
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MODEL_ARG;
    }
    if (messages_json == nullptr) {
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_MESSAGES_ARG;
    }
    if (out_rendered == nullptr) {
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_RENDERED_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_NULL_OUT_ERROR_ARG;
    }

    try {
        const char * tmpl_src = llama_model_chat_template(model, nullptr);
        if (tmpl_src == nullptr) {
            return LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_CHAT_TEMPLATE;
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        if (vocab == nullptr) {
            return LLAMA_RS_RENDER_CHAT_TEMPLATE_MODEL_HAS_NO_VOCAB;
        }

        std::string const bos_token = token_text_or_empty(vocab, llama_vocab_bos(vocab));
        std::string const eos_token = token_text_or_empty(vocab, llama_vocab_eos(vocab));

        common_chat_template const tmpl(tmpl_src, bos_token, eos_token);

        autoparser::generation_params params;
        params.add_generation_prompt = (add_generation_prompt != 0);
        params.enable_thinking = (enable_thinking != 0);
        params.is_inference = false;
        params.add_inference = false;
        params.mark_input = false;
        params.messages = nlohmann::ordered_json::parse(messages_json);

        std::string const rendered = common_chat_template_direct_apply(tmpl, params);

        *out_rendered = llama_rs_dup_string(rendered);
        if (*out_rendered == nullptr) {
            return LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_RENDER_CHAT_TEMPLATE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & ex) {
        *out_error = llama_rs_dup_string(std::string(ex.what()));
        if (*out_error == nullptr) {
            return LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string(std::string("unknown c++ exception"));
        if (*out_error == nullptr) {
            return LLAMA_RS_RENDER_CHAT_TEMPLATE_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_RENDER_CHAT_TEMPLATE_VENDORED_THREW_CXX_EXCEPTION;
    }
}
