#pragma once

#include <exception>
#include <new>

#include "wrapper_utils.h"

template <typename TStatus>
auto llama_rs_report_exception_message(
    char ** out_error,
    const char * message,
    TStatus reported_status,
    TStatus error_string_allocation_failed_status) noexcept -> TStatus {
    if (out_error == nullptr) {
        return reported_status;
    }
    *out_error = llama_rs_dup_string(message);
    if (*out_error == nullptr) {
        return error_string_allocation_failed_status;
    }
    return reported_status;
}

template <typename TStatus>
auto llama_rs_report_current_exception(
    char ** out_error,
    TStatus out_of_memory_status,
    TStatus threw_cxx_exception_status,
    TStatus error_string_allocation_failed_status) noexcept -> TStatus {
    try {
        throw;
    } catch (const std::bad_alloc &) {
        return out_of_memory_status;
    } catch (const std::exception & err) {
        return llama_rs_report_exception_message(
            out_error, err.what(), threw_cxx_exception_status, error_string_allocation_failed_status);
    } catch (...) {
        return llama_rs_report_exception_message(
            out_error, "unknown c++ exception", threw_cxx_exception_status, error_string_allocation_failed_status);
    }
}
