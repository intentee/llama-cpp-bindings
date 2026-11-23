#include "wrapper_mtmd.h"
#include "wrapper_fault_injection.h"
#include <llama.h>
#include <tools/mtmd/mtmd.h>
#include <tools/mtmd/mtmd-helper.h>
#include "wrapper_report_current_exception.h"

#include <cstddef>
#include <cstdint>
#include <gsl/span>

extern "C" auto llama_rs_mtmd_init_from_file(
    const char * mmproj_path,
    const struct llama_model * text_model,
    struct mtmd_context_params ctx_params,
    struct mtmd_context ** out_ctx,
    char ** out_error) -> llama_rs_mtmd_init_from_file_status {
    *out_error = nullptr;
    *out_ctx = nullptr;

    try {
        struct mtmd_context * ctx = mtmd_init_from_file(mmproj_path, text_model, ctx_params);
        if (ctx == nullptr) {
            return LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL;
        }
        *out_ctx = ctx;
        return LLAMA_RS_MTMD_INIT_FROM_FILE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_bitmap_init_from_file(
    const struct mtmd_context * ctx,
    const char * fname,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) -> llama_rs_mtmd_bitmap_init_from_file_status {
    *out_error = nullptr;
    *out_bitmap = nullptr;

    try {
        struct mtmd_helper_bitmap_wrapper const bitmap_wrapper =
            mtmd_helper_bitmap_init_from_file(ctx, fname, false, mtmd_helper_init_opt_default());
        struct mtmd_bitmap * bitmap = bitmap_wrapper.bitmap;
        if (bitmap == nullptr) {
            return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL;
        }
        *out_bitmap = bitmap;
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_bitmap_init_from_buf(
    const struct mtmd_context * ctx,
    const unsigned char * buf,
    size_t len,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) -> llama_rs_mtmd_bitmap_init_from_buf_status {
    *out_error = nullptr;
    *out_bitmap = nullptr;

    try {
        struct mtmd_helper_bitmap_wrapper const bitmap_wrapper =
            mtmd_helper_bitmap_init_from_buf(ctx, buf, len, false, mtmd_helper_init_opt_default());
        struct mtmd_bitmap * bitmap = bitmap_wrapper.bitmap;
        if (bitmap == nullptr) {
            return LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_RETURNED_NULL;
        }
        *out_bitmap = bitmap;
        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_tokenize(
    const struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_text * text,
    const struct mtmd_bitmap * const * bitmaps,
    size_t num_bitmaps,
    int32_t * out_undocumented_return_code,
    char ** out_error) -> llama_rs_mtmd_tokenize_status {
    *out_error = nullptr;
    *out_undocumented_return_code = 0;

    try {
        int32_t const result = mtmd_tokenize(ctx, output, text, bitmaps, num_bitmaps);
        switch (result) {
            case 0:
                return LLAMA_RS_MTMD_TOKENIZE_OK;
            case 1:
                return LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT;
            case 2:
                return LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE;
            default:
                *out_undocumented_return_code = result;
                return LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE;
        }
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_tokenize_from_parts(
    const struct mtmd_context * ctx,
    struct mtmd_input_chunks * output,
    const struct mtmd_input_part * const * parts,
    size_t num_parts,
    bool add_special,
    int32_t * out_undocumented_return_code,
    char ** out_error) -> llama_rs_mtmd_tokenize_from_parts_status {
    *out_error = nullptr;
    *out_undocumented_return_code = 0;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_MTMD_TOKENIZE_FROM_PARTS_THROWS);

        int32_t const result = mtmd_tokenize_from_parts(ctx, output, parts, num_parts, add_special);
        switch (result) {
            case 0:
                return LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK;
            case 1:
                return LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART;
            case 2:
                return LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE;
            default:
                *out_undocumented_return_code = result;
                return LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE;
        }
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_encode_chunk(
    struct mtmd_context * ctx,
    const struct mtmd_input_chunk * chunk,
    int32_t * out_llama_cpp_return_code,
    char ** out_error) -> llama_rs_mtmd_encode_chunk_status {
    *out_error = nullptr;
    *out_llama_cpp_return_code = 0;

    try {
        *out_llama_cpp_return_code = mtmd_encode_chunk(ctx, chunk);
        return LLAMA_RS_MTMD_ENCODE_CHUNK_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_eval_chunk_single(
    struct mtmd_context * ctx,
    struct llama_context * lctx,
    const struct mtmd_input_chunk * chunk,
    llama_pos n_past,
    llama_seq_id seq_id,
    int32_t n_batch,
    bool logits_last,
    llama_pos * out_new_n_past,
    int32_t * out_llama_cpp_return_code,
    char ** out_error) -> llama_rs_mtmd_eval_chunk_single_status {
    *out_error = nullptr;
    *out_llama_cpp_return_code = 0;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_MTMD_EVAL_CHUNK_SINGLE_THROWS);

        *out_llama_cpp_return_code = mtmd_helper_eval_chunk_single(
            ctx, lctx, chunk, n_past, seq_id, n_batch, logits_last, out_new_n_past);
        return LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_get_memory_usage(
    const char * mmproj_path,
    struct mtmd_context_params ctx_params,
    llama_rs_mtmd_device_memory * out_device_memory,
    size_t device_memory_capacity,
    size_t * out_device_memory_count,
    int32_t * out_image_max_tokens,
    bool * out_use_non_causal,
    char ** out_error) -> llama_rs_mtmd_get_memory_usage_status {
    *out_error = nullptr;
    *out_device_memory_count = 0;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_MTMD_GET_MEMORY_USAGE_THROWS);

        const mtmd_memory_usage usage = mtmd_get_memory_usage(mmproj_path, ctx_params);

        if (usage.backend_mem_usage.size() > device_memory_capacity) {
            return LLAMA_RS_MTMD_GET_MEMORY_USAGE_DEVICE_MEMORY_CAPACITY_EXCEEDED;
        }

        const gsl::span<llama_rs_mtmd_device_memory> device_memory(out_device_memory, device_memory_capacity);
        size_t device_memory_count = 0;

        for (const auto & [device, bytes] : usage.backend_mem_usage) {
            device_memory[device_memory_count] = llama_rs_mtmd_device_memory{device, bytes};
            device_memory_count += 1;
        }

        *out_device_memory_count = device_memory_count;
        *out_image_max_tokens = usage.image_max_tokens;
        *out_use_non_causal = usage.use_non_causal;

        return LLAMA_RS_MTMD_GET_MEMORY_USAGE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_GET_MEMORY_USAGE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_GET_MEMORY_USAGE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_bitmap_init(
    uint32_t width,
    uint32_t height,
    const unsigned char * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) -> llama_rs_mtmd_bitmap_init_status {
    *out_error = nullptr;
    *out_bitmap = nullptr;

    try {
        *out_bitmap = mtmd_bitmap_init(width, height, data);

        return LLAMA_RS_MTMD_BITMAP_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_BITMAP_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_bitmap_init_from_audio(
    size_t n_samples,
    const float * data,
    struct mtmd_bitmap ** out_bitmap,
    char ** out_error) -> llama_rs_mtmd_bitmap_init_from_audio_status {
    *out_error = nullptr;
    *out_bitmap = nullptr;

    try {
        *out_bitmap = mtmd_bitmap_init_from_audio(n_samples, data);

        return LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_input_chunks_init(
    struct mtmd_input_chunks ** out_chunks,
    char ** out_error) -> llama_rs_mtmd_input_chunks_init_status {
    *out_error = nullptr;
    *out_chunks = nullptr;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNKS_INIT_THROWS);

        *out_chunks = mtmd_input_chunks_init();

        return LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_input_chunk_copy(
    const struct mtmd_input_chunk * chunk,
    struct mtmd_input_chunk ** out_chunk,
    char ** out_error) -> llama_rs_mtmd_input_chunk_copy_status {
    *out_error = nullptr;
    *out_chunk = nullptr;

    try {
        *out_chunk = mtmd_input_chunk_copy(chunk);

        return LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_mtmd_input_chunk_get_type(const struct mtmd_input_chunk * chunk)
    -> mtmd_input_chunk_type {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_MTMD_INPUT_CHUNK_TYPE_UNKNOWN)) {
        return MTMD_INPUT_CHUNK_TYPE_COUNT;
    }

    return mtmd_input_chunk_get_type(chunk);
}
