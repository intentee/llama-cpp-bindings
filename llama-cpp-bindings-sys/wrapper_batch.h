#pragma once

#include <llama.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_batch_ext_init_status {
    LLAMA_RS_BATCH_EXT_INIT_OK = 0,
    LLAMA_RS_BATCH_EXT_INIT_NULL_CTX_ARG,
    LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_BATCH_ARG,
    LLAMA_RS_BATCH_EXT_INIT_NULL_OUT_ERROR_ARG,
    LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_batch_ext_init_status;

llama_rs_batch_ext_init_status llama_rs_batch_ext_init(
    struct llama_context * ctx,
    struct llama_batch_ext ** out_batch,
    char ** out_error);

typedef enum llama_rs_batch_ext_add_entry_status {
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_OK = 0,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_BATCH_ARG,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_SEQ_IDS_ARG,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_NULL_OUT_ERROR_ARG,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_EMPTY_SEQ_IDS,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_BATCH_FULL,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_TOKEN,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_INVALID_SEQUENCE_ID,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_RETURNED_UNKNOWN_CODE,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_POSITION_REJECTED,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_OUTPUT_REJECTED,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_BATCH_EXT_ADD_ENTRY_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_batch_ext_add_entry_status;

llama_rs_batch_ext_add_entry_status llama_rs_batch_ext_add_entry(
    struct llama_batch_ext * batch,
    llama_token token,
    llama_pos pos,
    const llama_seq_id * seq_ids,
    size_t n_seq_ids,
    bool output,
    int32_t * out_llama_cpp_return_code,
    char ** out_error);

#ifdef __cplusplus
}
#endif
