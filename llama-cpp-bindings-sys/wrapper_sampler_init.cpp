#include "wrapper_sampler_init.h"

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

namespace {

auto describe_init_failure(char ** out_error, const char * message) -> llama_rs_sampler_init_status {
    *out_error = llama_rs_dup_string(std::string(message));
    if (*out_error == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_ERROR_STRING_ALLOCATION_FAILED;
    }

    return LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION;
}

template <typename TInitializeSampler>
auto init_sampler(
    struct llama_sampler ** out_sampler,
    char ** out_error,
    TInitializeSampler initialize_sampler) -> llama_rs_sampler_init_status {
    if (out_sampler != nullptr) {
        *out_sampler = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_sampler == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_NULL_OUT_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_INIT_NULL_OUT_ERROR_ARG;
    }
    try {
        *out_sampler = initialize_sampler();
        if (*out_sampler == nullptr) {
            return LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_RETURNED_NULL;
        }

        return LLAMA_RS_SAMPLER_INIT_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_init_failure(out_error, err.what());
    } catch (...) {
        return describe_init_failure(out_error, "unknown c++ exception");
    }
}

auto describe_chain_add_failure(char ** out_error, const char * message)
    -> llama_rs_sampler_chain_add_status {
    *out_error = llama_rs_dup_string(std::string(message));
    if (*out_error == nullptr) {
        return LLAMA_RS_SAMPLER_CHAIN_ADD_ERROR_STRING_ALLOCATION_FAILED;
    }

    return LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION;
}

}  // namespace

extern "C" auto llama_rs_sampler_init(
    struct llama_sampler_i * iface,
    llama_sampler_context_t ctx,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [iface, ctx]() -> struct llama_sampler * { return llama_sampler_init(iface, ctx); });
}

extern "C" auto llama_rs_sampler_chain_init(
    struct llama_sampler_chain_params params,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [params]() -> struct llama_sampler * { return llama_sampler_chain_init(params); });
}

extern "C" auto llama_rs_sampler_init_greedy(
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, []() -> struct llama_sampler * { return llama_sampler_init_greedy(); });
}

extern "C" auto llama_rs_sampler_init_dist(
    uint32_t seed,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [seed]() -> struct llama_sampler * { return llama_sampler_init_dist(seed); });
}

extern "C" auto llama_rs_sampler_init_top_k(
    int32_t keep_count,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [keep_count]() -> struct llama_sampler * { return llama_sampler_init_top_k(keep_count); });
}

extern "C" auto llama_rs_sampler_init_top_p(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(
        out_sampler, out_error, [probability, min_keep]() -> struct llama_sampler * {
            return llama_sampler_init_top_p(probability, min_keep);
        });
}

extern "C" auto llama_rs_sampler_init_min_p(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(
        out_sampler, out_error, [probability, min_keep]() -> struct llama_sampler * {
            return llama_sampler_init_min_p(probability, min_keep);
        });
}

extern "C" auto llama_rs_sampler_init_typical(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(
        out_sampler, out_error, [probability, min_keep]() -> struct llama_sampler * {
            return llama_sampler_init_typical(probability, min_keep);
        });
}

extern "C" auto llama_rs_sampler_init_temp(
    float temperature,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [temperature]() -> struct llama_sampler * { return llama_sampler_init_temp(temperature); });
}

extern "C" auto llama_rs_sampler_init_temp_ext(
    float temperature,
    float delta,
    float exponent,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [temperature, delta, exponent]() -> struct llama_sampler * {
        return llama_sampler_init_temp_ext(temperature, delta, exponent);
    });
}

extern "C" auto llama_rs_sampler_init_xtc(
    float probability,
    float temperature,
    size_t min_keep,
    uint32_t seed,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [probability, temperature, min_keep, seed]() -> struct llama_sampler * {
        return llama_sampler_init_xtc(probability, temperature, min_keep, seed);
    });
}

extern "C" auto llama_rs_sampler_init_top_n_sigma(
    float sigma_multiplier,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [sigma_multiplier]() -> struct llama_sampler * {
        return llama_sampler_init_top_n_sigma(sigma_multiplier);
    });
}

extern "C" auto llama_rs_sampler_init_mirostat(
    int32_t n_vocab,
    uint32_t seed,
    float tau,
    float eta,
    int32_t candidate_count,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [n_vocab, seed, tau, eta, candidate_count]() -> struct llama_sampler * {
        return llama_sampler_init_mirostat(n_vocab, seed, tau, eta, candidate_count);
    });
}

extern "C" auto llama_rs_sampler_init_mirostat_v2(
    uint32_t seed,
    float tau,
    float eta,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [seed, tau, eta]() -> struct llama_sampler * {
        return llama_sampler_init_mirostat_v2(seed, tau, eta);
    });
}

extern "C" auto llama_rs_sampler_init_penalties(
    int32_t n_vocab,
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(
        out_sampler, out_error, [n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present]() -> struct llama_sampler * {
            return llama_sampler_init_penalties(
                n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present);
        });
}

extern "C" auto llama_rs_sampler_init_dry(
    const struct llama_vocab * vocab,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_penalty_last_n,
    const char ** seq_breakers,
    size_t num_breakers,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(
        out_sampler,
        out_error,
        [vocab, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers]()
            -> struct llama_sampler * {
            return llama_sampler_init_dry(
                vocab,
                dry_multiplier,
                dry_base,
                dry_allowed_length,
                dry_penalty_last_n,
                seq_breakers,
                num_breakers);
        });
}

extern "C" auto llama_rs_sampler_init_logit_bias(
    int32_t n_vocab,
    int32_t n_logit_bias,
    const llama_logit_bias * logit_bias,
    struct llama_sampler ** out_sampler,
    char ** out_error) -> llama_rs_sampler_init_status {
    return init_sampler(out_sampler, out_error, [n_vocab, n_logit_bias, logit_bias]() -> struct llama_sampler * {
        return llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias);
    });
}

extern "C" auto llama_rs_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * sampler,
    char ** out_error) -> llama_rs_sampler_chain_add_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (chain == nullptr) {
        return LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_CHAIN_ARG;
    }
    if (sampler == nullptr) {
        return LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_SAMPLER_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_OUT_ERROR_ARG;
    }
    try {
        llama_sampler_chain_add(chain, sampler);

        return LLAMA_RS_SAMPLER_CHAIN_ADD_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_chain_add_failure(out_error, err.what());
    } catch (...) {
        return describe_chain_add_failure(out_error, "unknown c++ exception");
    }
}
