#include "wrapper_context.h"

#include <cstdint>

#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-context.h"
#include "llama.cpp/src/llama-ext.h"
#include "llama.cpp/src/llama-model.h"

extern "C" auto llama_rs_context_decodes_batches_in_one_micro_batch(const struct llama_context * ctx) -> bool {
    return llama_get_memory(ctx) == nullptr || !ctx->get_cparams().causal_attn;
}

extern "C" auto llama_rs_context_embeddings_enabled(const struct llama_context * ctx) -> bool {
    return ctx->get_cparams().embeddings;
}

extern "C" auto llama_rs_context_embeddings_nextn_enabled(const struct llama_context * ctx) -> bool {
    return ctx->get_cparams().embeddings_nextn;
}

extern "C" auto llama_rs_context_embeddings_nextn_masked(const struct llama_context * ctx) -> bool {
    return ctx->get_cparams().embeddings_nextn_masked;
}

extern "C" void llama_rs_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked) {
    llama_set_embeddings_nextn(ctx, value, masked);
}

extern "C" auto llama_rs_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t token_index) -> float * {
    return llama_get_embeddings_nextn_ith(ctx, token_index);
}

extern "C" auto llama_rs_model_causal_attn(const struct llama_model * model) -> bool {
    return model->hparams.causal_attn;
}
