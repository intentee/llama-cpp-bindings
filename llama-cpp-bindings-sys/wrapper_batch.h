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

typedef struct llama_rs_batch_entry {
    llama_token token;
    llama_pos pos;
    size_t seq_ids_offset;
    size_t seq_ids_count;
    bool output;
} llama_rs_batch_entry;

typedef enum llama_rs_batch_ext_assign_entries_status {
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OK = 0,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_BATCH_ARG,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_ENTRIES_ARG,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_SEQ_IDS_ARG,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_LLAMA_CPP_RETURN_CODE_ARG,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_NULL_OUT_ERROR_ARG,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_SEQ_IDS_OUT_OF_RANGE,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_EMPTY_SEQ_IDS,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_BATCH_FULL,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_TOKEN,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_INVALID_SEQUENCE_ID,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_RETURNED_UNKNOWN_CODE,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_POSITION_REJECTED,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OUTPUT_REJECTED,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_batch_ext_assign_entries_status;

llama_rs_batch_ext_assign_entries_status llama_rs_batch_ext_assign_entries(
    struct llama_batch_ext * batch,
    const llama_rs_batch_entry * entries,
    size_t entries_count,
    const llama_seq_id * seq_ids,
    size_t seq_ids_count,
    int32_t * out_llama_cpp_return_code,
    char ** out_error);

#ifdef __cplusplus
}
#endif
