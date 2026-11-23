#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_adapter_lora;
struct llama_context;
struct llama_memory_i;
struct llama_model;

typedef enum llama_rs_memory_kind {
    LLAMA_RS_MEMORY_KIND_KV_CACHE = 0,
    LLAMA_RS_MEMORY_KIND_KV_CACHE_ISWA,
    LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA,
    LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA_ISWA,
    LLAMA_RS_MEMORY_KIND_KV_CACHE_MSA,
    LLAMA_RS_MEMORY_KIND_KV_CACHE_DSV4,
    LLAMA_RS_MEMORY_KIND_HYBRID,
    LLAMA_RS_MEMORY_KIND_HYBRID_ISWA,
    LLAMA_RS_MEMORY_KIND_HYBRID_IDX,
    LLAMA_RS_MEMORY_KIND_RECURRENT,
    LLAMA_RS_MEMORY_KIND_UNRECOGNIZED,
} llama_rs_memory_kind;

bool llama_rs_model_causal_attn(const struct llama_model * model);

uint32_t llama_rs_model_sliding_window(const struct llama_model * model);

bool llama_rs_context_kv_unified(const struct llama_context * ctx);

llama_rs_memory_kind llama_rs_memory_kind_of(const struct llama_memory_i * memory);

int32_t llama_rs_pooling_type(const struct llama_context * ctx);

int32_t llama_rs_model_rope_type(const struct llama_model * model);

bool llama_rs_context_produced_logits(struct llama_context * ctx);

float * llama_rs_get_logits_ith(struct llama_context * ctx, int32_t token_index);

int32_t llama_rs_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    float * scales);

#ifdef __cplusplus
}
#endif
