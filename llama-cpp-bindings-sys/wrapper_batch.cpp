#include "wrapper_batch.h"
#include "wrapper_report_current_exception.h"

#include <gsl/span>

extern "C" auto llama_rs_batch_ext_init(
    struct llama_context * ctx,
    struct llama_batch_ext ** out_batch,
    char ** out_error) -> llama_rs_batch_ext_init_status {
    if (out_batch != nullptr) {
        *out_batch = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_BATCH_EXT_INIT_NULL_CTX_ARG;
    }
    if (out_batch == nullptr) {
        return LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_BATCH_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_ERROR_ARG;
    }

    try {
        *out_batch = llama_batch_ext_init(ctx);

        return LLAMA_RS_BATCH_EXT_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_batch_ext_add_entry(
    struct llama_batch_ext * batch,
    llama_token token,
    llama_pos pos,
    const llama_seq_id * seq_ids,
    size_t n_seq_ids,
    bool output,
    int32_t * out_llama_cpp_return_code,
    char ** out_error) -> llama_rs_batch_ext_add_entry_status {
    if (out_llama_cpp_return_code != nullptr) {
        *out_llama_cpp_return_code = 0;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (batch == nullptr) {
        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_BATCH_ARG;
    }
    if (seq_ids == nullptr) {
        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_SEQ_IDS_ARG;
    }
    if (out_llama_cpp_return_code == nullptr) {
        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_ERROR_ARG;
    }
    if (n_seq_ids == 0) {
        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_EMPTY_SEQ_IDS;
    }

    try {
        const gsl::span<const llama_seq_id> seq_id_span(seq_ids, n_seq_ids);
        const int32_t index = llama_batch_ext_add_token(batch, seq_id_span[0], token);

        if (index < 0) {
            *out_llama_cpp_return_code = index;

            switch (index) {
                case -1:
                    return LLAMA_RS_BATCH_EXT_ADD_ENTRY_BATCH_FULL;
                case -2:
                    return LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_TOKEN;
                case -3:
                    return LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_SEQUENCE_ID;
                default:
                    return LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_RETURNED_UNKNOWN_CODE;
            }
        }

        for (const llama_seq_id seq_id : seq_id_span.subspan(1)) {
            if (!llama_batch_ext_add_seq(batch, index, seq_id)) {
                return LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_SEQUENCE_ID;
            }
        }

        if (!llama_batch_ext_set_pos(batch, index, &pos)) {
            return LLAMA_RS_BATCH_EXT_ADD_ENTRY_POSITION_REJECTED;
        }

        if (!llama_batch_ext_set_output_logits(batch, index, output)) {
            return LLAMA_RS_BATCH_EXT_ADD_ENTRY_OUTPUT_REJECTED;
        }

        return LLAMA_RS_BATCH_EXT_ADD_ENTRY_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_BATCH_EXT_ADD_ENTRY_ERROR_STRING_ALLOCATION_FAILED);
    }
}
