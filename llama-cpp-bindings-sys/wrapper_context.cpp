#include "wrapper_context.h"

#include <cstdint>

#include <llama.h>
#include <src/llama-context.h>
#include <src/llama-ext.h>
#include <src/llama-memory.h>
#include <src/llama-model.h>
#include "wrapper_report_current_exception.h"

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

extern "C" auto llama_rs_context_graph_produces_nextn_hidden_state(
    struct llama_context * ctx,
    bool * out_produces_nextn_hidden_state,
    char ** out_error) -> llama_rs_context_nextn_hidden_state_probe_status {
    if (out_produces_nextn_hidden_state != nullptr) {
        *out_produces_nextn_hidden_state = false;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_CTX_ARG;
    }
    if (out_produces_nextn_hidden_state == nullptr) {
        return LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_PRODUCES_NEXTN_HIDDEN_STATE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_NULL_OUT_ERROR_ARG;
    }
    try {
        constexpr uint32_t smallest_ubatch_tokens = 1;
        constexpr uint32_t smallest_ubatch_sequences = 1;
        constexpr uint32_t smallest_ubatch_outputs = 1;
        constexpr bool split_graph_without_allocating_buffers = true;

        llama_memory_context_ptr memory_context;
        llama_memory_t memory = llama_get_memory(ctx);
        if (memory != nullptr) {
            memory_context = memory->init_full();
        }
        if (ctx->graph_reserve(
                smallest_ubatch_tokens,
                smallest_ubatch_sequences,
                smallest_ubatch_outputs,
                memory_context.get(),
                split_graph_without_allocating_buffers) == nullptr) {
            return LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_FAILED_TO_RESERVE_GRAPH;
        }
        *out_produces_nextn_hidden_state = ctx->get_gf_res_reserve()->get_h_nextn() != nullptr;
        return LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CONTEXT_NEXTN_HIDDEN_STATE_PROBE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_model_causal_attn(const struct llama_model * model) -> bool {
    return model->hparams.causal_attn;
}
