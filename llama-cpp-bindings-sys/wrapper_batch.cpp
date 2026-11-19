#include "wrapper_batch.h"
#include "wrapper_fault_injection.h"
#include "wrapper_report_current_exception.h"

#include <llama.h>
#include <src/llama-ext.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gsl/span>

extern "C" auto llama_rs_batch_ext_init(
    struct llama_context * ctx,
    struct llama_batch_ext ** out_batch,
    char ** out_error) -> llama_rs_batch_ext_init_status {
    *out_batch = nullptr;
    *out_error = nullptr;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_BATCH_EXT_INIT_THROWS);
        *out_batch = llama_batch_ext_init(ctx);

        return LLAMA_RS_BATCH_EXT_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_BATCH_EXT_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_BATCH_EXT_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

namespace {

static_assert(static_cast<int>(LLAMA_RS_DECISION_ORDER_NONE) == static_cast<int>(LLAMA_DECISION_ORDER_NONE));
static_assert(
    static_cast<int>(LLAMA_RS_DECISION_ORDER_QUESTION_NOUL) == static_cast<int>(LLAMA_DECISION_ORDER_QUESTION_NOUL));
static_assert(
    static_cast<int>(LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE) ==
    static_cast<int>(LLAMA_DECISION_ORDER_QUESTION_CHOICE));
static_assert(
    static_cast<int>(LLAMA_RS_DECISION_ORDER_QUESTION_SCORE) == static_cast<int>(LLAMA_DECISION_ORDER_QUESTION_SCORE));
static_assert(static_cast<int>(LLAMA_RS_DECISION_ORDER_OPTION) == static_cast<int>(LLAMA_DECISION_ORDER_OPTION));

auto add_entry(
    struct llama_batch_ext * batch,
    const llama_rs_batch_entry & entry,
    const gsl::span<const llama_seq_id> seq_ids,
    int32_t & out_llama_cpp_return_code) -> llama_rs_batch_ext_assign_entries_status {
    const int32_t index = llama_batch_ext_add_token(batch, seq_ids[0], entry.token);

    if (index < 0) {
        out_llama_cpp_return_code = index;

        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_REFUSED_TOKEN;
    }

    const gsl::span<const llama_seq_id> additional_seq_ids = seq_ids.subspan(1);
    const bool every_seq_id_added = std::all_of(
        additional_seq_ids.begin(), additional_seq_ids.end(), [&](const llama_seq_id seq_id) -> bool {
            return llama_batch_ext_add_seq(batch, index, seq_id);
        });

    if (!every_seq_id_added) {
        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_REFUSED_ADDITIONAL_SEQUENCE_ID;
    }

    if (!llama_batch_ext_set_pos(batch, index, &entry.pos)) {
        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_POSITION_REJECTED;
    }

    if (!llama_batch_ext_set_output_logits(batch, index, entry.output)) {
        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OUTPUT_REJECTED;
    }

    if (!llama_batch_ext_set_decision_order(batch, index, static_cast<llama_decision_order>(entry.decision_order))) {
        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_DECISION_ORDER_REJECTED;
    }

    return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OK;
}

}  // namespace

extern "C" auto llama_rs_batch_ext_assign_entries(
    struct llama_batch_ext * batch,
    const llama_rs_batch_entry * entries,
    size_t entries_count,
    const llama_seq_id * seq_ids,
    size_t seq_ids_count,
    int32_t * out_llama_cpp_return_code,
    char ** out_error) -> llama_rs_batch_ext_assign_entries_status {
    *out_llama_cpp_return_code = 0;
    *out_error = nullptr;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_BATCH_EXT_ASSIGN_ENTRIES_THROWS);
        llama_batch_ext_clear(batch);

        const gsl::span<const llama_rs_batch_entry> entry_span(entries, entries_count);
        const gsl::span<const llama_seq_id> seq_id_span(seq_ids, seq_ids_count);

        for (const llama_rs_batch_entry & entry : entry_span) {
            const llama_rs_batch_ext_assign_entries_status status = add_entry(
                batch,
                entry,
                seq_id_span.subspan(entry.seq_ids_offset, entry.seq_ids_count),
                *out_llama_cpp_return_code);

            if (status != LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OK) {
                return status;
            }
        }

        return LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_BATCH_EXT_ASSIGN_ENTRIES_ERROR_STRING_ALLOCATION_FAILED);
    }
}
