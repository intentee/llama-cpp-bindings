#include "wrapper_context.h"
#include "wrapper_fault_injection.h"

#include <cstddef>
#include <cstdint>
#include <typeinfo>

#include <llama.h>
#include <src/llama-context.h>
#include <src/llama-kv-cache-dsa-iswa.h>
#include <src/llama-kv-cache-dsa.h>
#include <src/llama-kv-cache-dsv4.h>
#include <src/llama-kv-cache-iswa.h>
#include <src/llama-kv-cache-msa.h>
#include <src/llama-kv-cache.h>
#include <src/llama-memory-hybrid-idx.h>
#include <src/llama-memory-hybrid-iswa.h>
#include <src/llama-memory-hybrid.h>
#include <src/llama-memory-recurrent.h>
#include <src/llama-memory.h>
#include <src/llama-model.h>

extern "C" auto llama_rs_model_causal_attn(const struct llama_model * model) -> bool {
    return model->hparams.causal_attn;
}

extern "C" auto llama_rs_model_sliding_window(const struct llama_model * model) -> uint32_t {
    return model->hparams.n_swa;
}

extern "C" auto llama_rs_context_kv_unified(const struct llama_context * ctx) -> bool {
    return ctx->get_cparams().kv_unified;
}

extern "C" auto llama_rs_memory_kind_of(const struct llama_memory_i * memory) -> llama_rs_memory_kind {
    const std::type_info & memory_type = typeid(*memory);

    if (memory_type == typeid(llama_kv_cache)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE;
    }

    if (memory_type == typeid(llama_kv_cache_iswa)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE_ISWA;
    }

    if (memory_type == typeid(llama_kv_cache_dsa)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA;
    }

    if (memory_type == typeid(llama_kv_cache_dsa_iswa)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA_ISWA;
    }

    if (memory_type == typeid(llama_kv_cache_msa)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE_MSA;
    }

    if (memory_type == typeid(llama_kv_cache_dsv4)) {
        return LLAMA_RS_MEMORY_KIND_KV_CACHE_DSV4;
    }

    if (memory_type == typeid(llama_memory_hybrid)) {
        return LLAMA_RS_MEMORY_KIND_HYBRID;
    }

    if (memory_type == typeid(llama_memory_hybrid_iswa)) {
        return LLAMA_RS_MEMORY_KIND_HYBRID_ISWA;
    }

    if (memory_type == typeid(llama_memory_hybrid_idx)) {
        return LLAMA_RS_MEMORY_KIND_HYBRID_IDX;
    }

    if (memory_type == typeid(llama_memory_recurrent)) {
        return LLAMA_RS_MEMORY_KIND_RECURRENT;
    }

    return LLAMA_RS_MEMORY_KIND_UNRECOGNIZED;
}

extern "C" auto llama_rs_pooling_type(const struct llama_context * ctx) -> int32_t {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_POOLING_TYPE_UNKNOWN)) {
        return LLAMA_POOLING_TYPE_RANK + 1;
    }

    return llama_pooling_type(ctx);
}

extern "C" auto llama_rs_model_rope_type(const struct llama_model * model) -> int32_t {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_ROPE_TYPE_UNKNOWN)) {
        return LLAMA_ROPE_TYPE_IMROPE + 1;
    }

    return llama_model_rope_type(model);
}

extern "C" auto llama_rs_context_produced_logits(struct llama_context * ctx) -> bool {
    return llama_get_logits(ctx) != nullptr;
}

extern "C" auto llama_rs_get_logits_ith(struct llama_context * ctx, int32_t token_index) -> float * {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_GET_LOGITS_ITH_RETURNS_NULL)) {
        return nullptr;
    }

    return llama_get_logits_ith(ctx, token_index);
}

extern "C" auto llama_rs_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    float * scales) -> int32_t {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_SET_ADAPTERS_LORA_FAILS)) {
        return -1;
    }

    return llama_set_adapters_lora(ctx, adapters, n_adapters, scales);
}
