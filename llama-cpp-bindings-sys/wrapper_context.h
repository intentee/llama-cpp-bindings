#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_adapter_lora;
struct llama_context;
struct llama_model;

bool llama_rs_model_causal_attn(const struct llama_model * model);

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
