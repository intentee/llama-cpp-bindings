#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_context;
struct llama_model;

bool llama_rs_context_decodes_batches_in_one_micro_batch(const struct llama_context * ctx);

size_t llama_rs_context_embedding_row_length(const struct llama_context * ctx);

void llama_rs_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);

float * llama_rs_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t token_index);

bool llama_rs_model_causal_attn(const struct llama_model * model);

#ifdef __cplusplus
}
#endif
