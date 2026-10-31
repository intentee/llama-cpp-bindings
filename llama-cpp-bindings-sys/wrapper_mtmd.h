#pragma once

#include <llama.h>
#include <tools/mtmd/mtmd.h>
#include <tools/mtmd/mtmd-helper.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_mtmd_init_from_file_status {
    LLAMA_RS_MTMD_INIT_FROM_FILE_OK = 0,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG,
    LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL,
    LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_init_from_file_status;

llama_rs_mtmd_init_from_file_status llama_rs_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    struct mtmd_context_params ctx_params,
    struct mtmd_context ** out_ctx,
    char ** out_error);

typedef enum llama_rs_mtmd_bitmap_init_from_file_status {
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK = 0,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_FNAME_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_OUT_BITMAP_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_bitmap_init_from_file_status;

llama_rs_mtmd_bitmap_init_from_file_status llama_rs_mtmd_bitmap_init_from_file(
    const struct mtmd_context * ctx,
    const char * fname,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

typedef enum llama_rs_mtmd_bitmap_init_from_buf_status {
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_OK = 0,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_NULL_CTX_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_NULL_BUF_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_NULL_OUT_BITMAP_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_RETURNED_NULL,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_bitmap_init_from_buf_status;

llama_rs_mtmd_bitmap_init_from_buf_status llama_rs_mtmd_bitmap_init_from_buf(
    const struct mtmd_context * ctx,
    const unsigned char * buf,
    size_t len,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

typedef enum llama_rs_mtmd_tokenize_status {
    LLAMA_RS_MTMD_TOKENIZE_OK = 0,
    LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG,
    LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO,
    LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
    LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
    LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
    LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_tokenize_status;

llama_rs_mtmd_tokenize_status llama_rs_mtmd_tokenize(
    const struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap * const * bitmaps,
    size_t num_bitmaps,
    int32_t * out_undocumented_return_code,
    char ** out_error);

typedef enum llama_rs_mtmd_tokenize_from_parts_status {
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK = 0,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_CTX_ARG,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_OUTPUT_ARG,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PARTS_ARG_WHEN_NUM_PARTS_NONZERO,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PART_ELEMENT,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_tokenize_from_parts_status;

llama_rs_mtmd_tokenize_from_parts_status llama_rs_mtmd_tokenize_from_parts(
    const struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_part * const * parts,
    size_t num_parts,
    bool add_special,
    int32_t * out_undocumented_return_code,
    char ** out_error);

typedef enum llama_rs_mtmd_encode_chunk_status {
    LLAMA_RS_MTMD_ENCODE_CHUNK_OK = 0,
    LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG,
    LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG,
    LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_RETURNED_NONZERO_CODE,
    LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_encode_chunk_status;

llama_rs_mtmd_encode_chunk_status llama_rs_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_llama_cpp_return_code,
    char ** out_error);

typedef enum llama_rs_mtmd_eval_chunk_single_status {
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK = 0,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_RETURNED_NONZERO_CODE,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_eval_chunk_single_status;

llama_rs_mtmd_eval_chunk_single_status llama_rs_mtmd_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_llama_cpp_return_code,
    char ** out_error);

typedef struct llama_rs_mtmd_device_memory {
    ggml_backend_dev_t device;
    size_t bytes;
} llama_rs_mtmd_device_memory;

typedef enum llama_rs_mtmd_get_memory_usage_status {
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK = 0,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_MMPROJ_PATH_ARG,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_ARG,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_DEVICE_MEMORY_COUNT_ARG,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_IMAGE_MAX_TOKENS_ARG,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_NULL_OUT_USE_NON_CAUSAL_ARG,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_get_memory_usage_status;

llama_rs_mtmd_get_memory_usage_status llama_rs_mtmd_get_memory_usage(
    const char * mmproj_path,
    struct mtmd_context_params ctx_params,
    llama_rs_mtmd_device_memory * out_device_memory,
    size_t device_memory_capacity,
    size_t * out_device_memory_count,
    int32_t * out_image_max_tokens,
    bool * out_use_non_causal,
    char ** out_error);

typedef enum llama_rs_mtmd_bitmap_init_status {
    LLAMA_RS_MTMD_BITMAP_INIT_OK = 0,
    LLAMA_RS_MTMD_BITMAP_INIT_NULL_DATA_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_NULL_OUT_BITMAP_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_bitmap_init_status;

llama_rs_mtmd_bitmap_init_status llama_rs_mtmd_bitmap_init(
    uint32_t width,
    uint32_t height,
    const unsigned char * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

typedef enum llama_rs_mtmd_bitmap_init_from_audio_status {
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_OK = 0,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_NULL_DATA_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_NULL_OUT_BITMAP_ARG,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_bitmap_init_from_audio_status;

llama_rs_mtmd_bitmap_init_from_audio_status llama_rs_mtmd_bitmap_init_from_audio(
    size_t n_samples,
    const float * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error);

typedef enum llama_rs_mtmd_input_chunks_init_status {
    LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK = 0,
    LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_NULL_OUT_CHUNKS_ARG,
    LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_input_chunks_init_status;

llama_rs_mtmd_input_chunks_init_status llama_rs_mtmd_input_chunks_init(
    struct mtmd_input_chunks ** out_chunks,
    char ** out_error);

typedef enum llama_rs_mtmd_input_chunk_copy_status {
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK = 0,
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_CHUNK_ARG,
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_OUT_CHUNK_ARG,
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_mtmd_input_chunk_copy_status;

llama_rs_mtmd_input_chunk_copy_status llama_rs_mtmd_input_chunk_copy(
    const struct mtmd_input_chunk * chunk,
    struct mtmd_input_chunk ** out_chunk,
    char ** out_error);

const struct mtmd_input_chunk * llama_rs_mtmd_input_chunks_get(
    const struct mtmd_input_chunks * chunks,
    size_t index);

enum mtmd_input_chunk_type llama_rs_mtmd_input_chunk_get_type(const struct mtmd_input_chunk * chunk);

#ifdef __cplusplus
}
#endif
