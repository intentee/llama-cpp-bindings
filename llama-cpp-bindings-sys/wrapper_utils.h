#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus

#include <cstring>
#include <new>
#include <string>

static inline char * llama_rs_dup_string(const std::string & value) {
    char * buffer = new (std::nothrow) char[value.size() + 1];
    if (!buffer) {
        return nullptr;
    }
    std::memcpy(buffer, value.data(), value.size());
    buffer[value.size()] = '\0';
    return buffer;
}

#endif
