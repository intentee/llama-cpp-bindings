#include "wrapper_context.h"
#include "wrapper_fault_injection.h"

#include <cstddef>
#include <cstdint>

#include <llama.h>
#include <src/llama-model.h>

extern "C" auto llama_rs_model_causal_attn(const struct llama_model * model) -> bool {
    return model->hparams.causal_attn;
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
