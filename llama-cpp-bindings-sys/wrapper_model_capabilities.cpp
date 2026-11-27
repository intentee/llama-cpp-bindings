#include "wrapper_model_capabilities.h"
#include "wrapper_chat_templates.h"
#include "wrapper_chat_templates_handle.h"
#include "wrapper_report_current_exception.h"

#include <common/chat.h>
#include <common/common.h>
#include <common/json.h>
#include <llama.h>
#include <server-common.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t METADATA_VALUE_BUFFER_SIZE = 128;
constexpr size_t ARCHITECTURE_BUFFER_SIZE = 256;

auto meta_value(const llama_model * model, llama_model_meta_key key, std::string & out_value) -> bool {
    std::vector<char> buffer(METADATA_VALUE_BUFFER_SIZE, '\0');

    if (llama_model_meta_val_str(model, llama_model_meta_key_str(key), buffer.data(), buffer.size()) <= 0) {
        return false;
    }

    out_value.assign(buffer.data());

    return true;
}

auto parsed_int32(const std::string & text, int32_t & out_value) -> bool {
    try {
        out_value = std::stoi(text);

        return true;
    } catch (const std::invalid_argument &) {
        return false;
    } catch (const std::out_of_range &) {
        return false;
    }
}

auto parsed_float(const std::string & text, float & out_value) -> bool {
    try {
        out_value = std::stof(text);

        return true;
    } catch (const std::invalid_argument &) {
        return false;
    } catch (const std::out_of_range &) {
        return false;
    }
}

void read_int32(const llama_model * model, llama_model_meta_key key, int32_t & out_value) {
    std::string text;
    int32_t value = 0;

    if (meta_value(model, key, text) && parsed_int32(text, value)) {
        out_value = value;
    }
}

void read_float(const llama_model * model, llama_model_meta_key key, float & out_value) {
    std::string text;

    if (!meta_value(model, key, text)) {
        return;
    }

    float value = 0.0F;

    if (parsed_float(text, value)) {
        out_value = value;
    }
}
}  // namespace

extern "C" auto llama_rs_model_declared_pooling_type(const struct llama_model * model) -> int32_t {
    std::vector<char> architecture(ARCHITECTURE_BUFFER_SIZE, '\0');

    if (llama_model_meta_val_str(model, "general.architecture", architecture.data(), architecture.size()) <= 0) {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }

    const std::string pooling_key = std::string(architecture.data()) + ".pooling_type";
    std::vector<char> pooling(METADATA_VALUE_BUFFER_SIZE, '\0');

    if (llama_model_meta_val_str(model, pooling_key.c_str(), pooling.data(), pooling.size()) <= 0) {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }

    int32_t pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;

    if (!parsed_int32(std::string(pooling.data()), pooling_type)) {
        return LLAMA_POOLING_TYPE_UNSPECIFIED;
    }

    return pooling_type;
}

extern "C" auto llama_rs_model_outputs_text(const struct llama_model * model, bool projector_chats, bool projector_speaks)
    -> bool {
    const std::vector<std::string> modalities = server_model_output_modalities(model, projector_chats, projector_speaks);

    return std::find(modalities.begin(), modalities.end(), "text") != modalities.end();
}

extern "C" auto llama_rs_chat_templates_capabilities_of(
    llama_rs_chat_templates_handle templates,
    llama_rs_chat_templates_capabilities * out_capabilities,
    char ** out_error) -> llama_rs_chat_templates_capabilities_status {
    *out_error = nullptr;

    try {
        const common_json caps = common_chat_templates_get_caps(templates->templates.get());

        *out_capabilities = llama_rs_chat_templates_capabilities{
            caps.at("supports_parallel_tool_calls").get<bool>(),
            caps.at("supports_system_role").get<bool>(),
            common_chat_templates_support_enable_thinking(templates->templates.get()),
            caps.at("supports_tool_calls").get<bool>(),
            caps.at("supports_tools").get<bool>(),
        };

        return LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_TEMPLATES_CAPABILITIES_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_model_sampling_defaults(const struct llama_model * model) -> llama_rs_sampling_defaults {
    common_params_sampling sampling;

    read_int32(model, LLAMA_MODEL_META_KEY_SAMPLING_TOP_K, sampling.top_k);
    read_float(model, LLAMA_MODEL_META_KEY_SAMPLING_TOP_P, sampling.top_p);
    read_float(model, LLAMA_MODEL_META_KEY_SAMPLING_MIN_P, sampling.min_p);
    read_float(model, LLAMA_MODEL_META_KEY_SAMPLING_TEMP, sampling.temp);
    read_int32(model, LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N, sampling.penalty_last_n);
    read_float(model, LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT, sampling.penalty_repeat);

    return llama_rs_sampling_defaults{
        sampling.min_p,
        sampling.penalty_freq,
        sampling.penalty_last_n,
        sampling.penalty_present,
        sampling.penalty_repeat,
        sampling.temp,
        sampling.top_k,
        sampling.top_p,
    };
}
