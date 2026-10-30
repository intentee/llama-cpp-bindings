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

typedef enum llama_rs_context_nextn_hidden_state_probe_status {
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_OK = 0,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_CTX_ARG,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_PRODUCES_NEXTN_HIDDEN_STATE_ARG,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_FAILED_TO_RESERVE_GRAPH,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_context_nextn_hidden_state_probe_status;

bool llama_rs_context_embeddings_enabled(const struct llama_context * ctx);

bool llama_rs_context_embeddings_nextn_enabled(const struct llama_context * ctx);

bool llama_rs_context_embeddings_nextn_masked(const struct llama_context * ctx);

void llama_rs_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

float * llama_rs_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t token_index);

llama_rs_context_nextn_hidden_state_probe_status llama_rs_context_graph_produces_nextn_hidden_state(
    struct llama_context * ctx,
    bool * out_produces_nextn_hidden_state,
    char ** out_error);

bool llama_rs_model_causal_attn(const struct llama_model * model);

float * llama_rs_get_logits_ith(struct llama_context * ctx, int32_t token_index);

int32_t llama_rs_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    float * scales);

#ifdef __cplusplus
}
#endif
