#pragma once

#include "llama.cpp/include/llama.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_state_data_status {
    LLAMA_RS_STATE_DATA_OK = 0,
    LLAMA_RS_STATE_DATA_NULL_CTX_ARG,
    LLAMA_RS_STATE_DATA_NULL_BUFFER_ARG,
    LLAMA_RS_STATE_DATA_NULL_OUT_BYTE_COUNT_ARG,
    LLAMA_RS_STATE_DATA_NULL_OUT_ERROR_ARG,
    LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY,
    LLAMA_RS_STATE_DATA_VENDORED_THREW_CXX_EXCEPTION,
} llama_rs_state_data_status;

llama_rs_state_data_status llama_rs_state_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    size_t * out_byte_count,
    char ** out_error);

llama_rs_state_data_status llama_rs_state_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    size_t * out_byte_count,
    char ** out_error);

llama_rs_state_data_status llama_rs_state_seq_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    llama_seq_id seq_id,
    llama_state_seq_flags flags,
    size_t * out_byte_count,
    char ** out_error);

llama_rs_state_data_status llama_rs_state_seq_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    llama_seq_id dest_seq_id,
    llama_state_seq_flags flags,
    size_t * out_byte_count,
    char ** out_error);

#ifdef __cplusplus
}
#endif
