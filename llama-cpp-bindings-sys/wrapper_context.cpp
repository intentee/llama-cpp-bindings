#include "wrapper_context.h"

#include "llama.cpp/include/llama.h"
#include "llama.cpp/src/llama-context.h"

extern "C" auto llama_rs_context_decodes_batches_in_one_micro_batch(const struct llama_context * ctx) -> bool {
    return llama_get_memory(ctx) == nullptr || !ctx->get_cparams().causal_attn;
}
