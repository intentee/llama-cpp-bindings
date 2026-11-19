#include "wrapper_state.h"

#include <llama.h>
#include "wrapper_report_current_exception.h"

#include <cstddef>
#include <cstdint>

extern "C" auto llama_rs_state_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    *out_error = nullptr;
    *out_byte_count = 0;

    try {
        *out_byte_count = llama_state_get_data(ctx, dst, size);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_state_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    *out_error = nullptr;
    *out_byte_count = 0;

    try {
        *out_byte_count = llama_state_set_data(ctx, src, size);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_state_seq_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    llama_seq_id seq_id,
    llama_state_seq_flags flags,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    *out_error = nullptr;
    *out_byte_count = 0;

    try {
        *out_byte_count = llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_state_seq_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    llama_seq_id dest_seq_id,
    llama_state_seq_flags flags,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    *out_error = nullptr;
    *out_byte_count = 0;

    try {
        *out_byte_count = llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_STATE_DATA_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED);
    }
}
