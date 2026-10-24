#pragma once

#include "llama.cpp/include/llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_sampler_init_status {
    LLAMA_RS_SAMPLER_INIT_OK = 0,
    LLAMA_RS_SAMPLER_INIT_NULL_OUT_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_INIT_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_RETURNED_NULL,
    LLAMA_RS_SAMPLER_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_SAMPLER_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_sampler_init_status;

llama_rs_sampler_init_status llama_rs_sampler_init(
    struct llama_sampler_i * iface,
    llama_sampler_context_t ctx,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_chain_init(
    struct llama_sampler_chain_params params,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_greedy(
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_dist(
    uint32_t seed,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_top_k(
    int32_t keep_count,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_top_p(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_min_p(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_typical(
    float probability,
    size_t min_keep,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_temp(
    float temperature,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_temp_ext(
    float temperature,
    float delta,
    float exponent,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_xtc(
    float probability,
    float temperature,
    size_t min_keep,
    uint32_t seed,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_top_n_sigma(
    float sigma_multiplier,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_mirostat(
    int32_t n_vocab,
    uint32_t seed,
    float tau,
    float eta,
    int32_t candidate_count,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_mirostat_v2(
    uint32_t seed,
    float tau,
    float eta,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_penalties(
    int32_t n_vocab,
    int32_t penalty_last_n,
    float penalty_repeat,
    float penalty_freq,
    float penalty_present,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_dry(
    const struct llama_vocab * vocab,
    float dry_multiplier,
    float dry_base,
    int32_t dry_allowed_length,
    int32_t dry_penalty_last_n,
    const char ** seq_breakers,
    size_t num_breakers,
    struct llama_sampler ** out_sampler,
    char ** out_error);

llama_rs_sampler_init_status llama_rs_sampler_init_logit_bias(
    int32_t n_vocab,
    int32_t n_logit_bias,
    const llama_logit_bias * logit_bias,
    struct llama_sampler ** out_sampler,
    char ** out_error);

typedef enum llama_rs_sampler_chain_add_status {
    LLAMA_RS_SAMPLER_CHAIN_ADD_OK = 0,
    LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_CHAIN_ARG,
    LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_SAMPLER_ARG,
    LLAMA_RS_SAMPLER_CHAIN_ADD_NULL_OUT_ERROR_ARG,
    LLAMA_RS_SAMPLER_CHAIN_ADD_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_SAMPLER_CHAIN_ADD_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_sampler_chain_add_status;

llama_rs_sampler_chain_add_status llama_rs_sampler_chain_add(
    struct llama_sampler * chain,
    struct llama_sampler * sampler,
    char ** out_error);

#ifdef __cplusplus
}
#endif
