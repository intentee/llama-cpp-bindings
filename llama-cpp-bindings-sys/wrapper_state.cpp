#include "wrapper_state.h"

#include "llama.cpp/include/llama.h"
#include "wrapper_utils.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <string>

namespace {

auto describe_failure(char ** out_error, const char * message) -> llama_rs_state_data_status {
    *out_error = llama_rs_dup_string(std::string(message));
    if (*out_error == nullptr) {
        return LLAMA_RS_STATE_DATA_ERROR_STRING_ALLOCATION_FAILED;
    }

    return LLAMA_RS_STATE_DATA_VENDORED_THREW_CXX_EXCEPTION;
}

auto validate_arguments(
    const struct llama_context * ctx,
    const void * buffer,
    const size_t * out_byte_count,
    char * const * out_error) -> llama_rs_state_data_status {
    if (out_error == nullptr) {
        return LLAMA_RS_STATE_DATA_NULL_OUT_ERROR_ARG;
    }
    if (ctx == nullptr) {
        return LLAMA_RS_STATE_DATA_NULL_CTX_ARG;
    }
    if (buffer == nullptr) {
        return LLAMA_RS_STATE_DATA_NULL_BUFFER_ARG;
    }
    if (out_byte_count == nullptr) {
        return LLAMA_RS_STATE_DATA_NULL_OUT_BYTE_COUNT_ARG;
    }

    return LLAMA_RS_STATE_DATA_OK;
}

}  // namespace

extern "C" auto llama_rs_state_get_data(
    struct llama_context * ctx,
    uint8_t * dst,
    size_t size,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_byte_count != nullptr) {
        *out_byte_count = 0;
    }

    const llama_rs_state_data_status rejected =
        validate_arguments(ctx, dst, out_byte_count, out_error);
    if (rejected != LLAMA_RS_STATE_DATA_OK) {
        return rejected;
    }

    try {
        *out_byte_count = llama_state_get_data(ctx, dst, size);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_failure(out_error, err.what());
    } catch (...) {
        return describe_failure(out_error, "unknown c++ exception");
    }
}

extern "C" auto llama_rs_state_set_data(
    struct llama_context * ctx,
    const uint8_t * src,
    size_t size,
    size_t * out_byte_count,
    char ** out_error) -> llama_rs_state_data_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_byte_count != nullptr) {
        *out_byte_count = 0;
    }

    const llama_rs_state_data_status rejected =
        validate_arguments(ctx, src, out_byte_count, out_error);
    if (rejected != LLAMA_RS_STATE_DATA_OK) {
        return rejected;
    }

    try {
        *out_byte_count = llama_state_set_data(ctx, src, size);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_failure(out_error, err.what());
    } catch (...) {
        return describe_failure(out_error, "unknown c++ exception");
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
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_byte_count != nullptr) {
        *out_byte_count = 0;
    }

    const llama_rs_state_data_status rejected =
        validate_arguments(ctx, dst, out_byte_count, out_error);
    if (rejected != LLAMA_RS_STATE_DATA_OK) {
        return rejected;
    }

    try {
        *out_byte_count = llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_failure(out_error, err.what());
    } catch (...) {
        return describe_failure(out_error, "unknown c++ exception");
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
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (out_byte_count != nullptr) {
        *out_byte_count = 0;
    }

    const llama_rs_state_data_status rejected =
        validate_arguments(ctx, src, out_byte_count, out_error);
    if (rejected != LLAMA_RS_STATE_DATA_OK) {
        return rejected;
    }

    try {
        *out_byte_count = llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags);

        return LLAMA_RS_STATE_DATA_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_STATE_DATA_VENDORED_OUT_OF_MEMORY;
    } catch (const std::exception & err) {
        return describe_failure(out_error, err.what());
    } catch (...) {
        return describe_failure(out_error, "unknown c++ exception");
    }
}
