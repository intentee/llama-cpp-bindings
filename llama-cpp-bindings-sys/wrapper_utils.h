#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus

#include <cstring>
#include <new>
#include <string>

static inline char * llama_rs_dup_bytes(const char * data, size_t size) noexcept {
    char * buffer = new (std::nothrow) char[size + 1];
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, data, size);
    buffer[size] = '\0';
    return buffer;
}

static inline char * llama_rs_dup_string(const std::string & value) noexcept {
    return llama_rs_dup_bytes(value.data(), value.size());
}

static inline char * llama_rs_dup_string(const char * value) noexcept {
    return llama_rs_dup_bytes(value, std::strlen(value));
}

#endif
