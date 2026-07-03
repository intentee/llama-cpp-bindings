#include "wrapper_gbnf.h"

#include "alloc_fault.h"
#include "wrapper_utils.h"

#include "llama-grammar.h"
#include "unicode.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

struct fault_region {
    fault_region() {
        gbnf_alloc_fault_region_enter();
    }

    ~fault_region() {
        gbnf_alloc_fault_region_exit();
    }

    fault_region(const fault_region &) = delete;
    fault_region & operator=(const fault_region &) = delete;
    fault_region(fault_region &&) = delete;
    fault_region & operator=(fault_region &&) = delete;
};

} // namespace

extern "C" {

llama_rs_gbnf_parse_status llama_rs_gbnf_parse(
    const char * grammar_str,
    const char * grammar_root,
    struct llama_grammar ** out_grammar) {
    try {
        llama_grammar_parser parser(nullptr);

        if (!parser.parse(grammar_str)) {
            return LLAMA_RS_GBNF_PARSE_SYNTAX_ERROR;
        }

        if (parser.rules.empty()) {
            return LLAMA_RS_GBNF_PARSE_EMPTY_RULE_SET;
        }

        const auto root_iterator = parser.symbol_ids.find(grammar_root);

        if (root_iterator == parser.symbol_ids.end()) {
            return LLAMA_RS_GBNF_PARSE_ROOT_SYMBOL_MISSING;
        }

        const fault_region region;
        auto grammar_rules = parser.c_rules();
        llama_grammar * grammar = llama_grammar_init_impl(
            nullptr,
            grammar_rules.data(),
            grammar_rules.size(),
            root_iterator->second);

        if (grammar == nullptr) {
            return LLAMA_RS_GBNF_PARSE_LEFT_RECURSION;
        }

        *out_grammar = grammar;

        return LLAMA_RS_GBNF_PARSE_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            nullptr,
            LLAMA_RS_GBNF_PARSE_ALLOCATION_FAILED,
            LLAMA_RS_GBNF_PARSE_THREW_CXX_EXCEPTION);
    }
}

llama_rs_gbnf_runtime_status llama_rs_gbnf_accept_str(
    struct llama_grammar * grammar,
    const char * piece,
    size_t piece_len) {
    try {
        const fault_region region;
        const std::string input(piece, piece_len);
        const std::vector<uint32_t> code_points = unicode_cpts_from_utf8(input);
        auto & stacks = llama_grammar_get_stacks(grammar);

        for (const uint32_t code_point : code_points) {
            if (stacks.empty()) {
                break;
            }

            llama_grammar_accept(grammar, code_point);
        }

        return LLAMA_RS_GBNF_RUNTIME_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            nullptr,
            LLAMA_RS_GBNF_RUNTIME_ALLOCATION_FAILED,
            LLAMA_RS_GBNF_RUNTIME_THREW_CXX_EXCEPTION);
    }
}

bool llama_rs_gbnf_is_accepting(struct llama_grammar * grammar) noexcept {
    auto & stacks = llama_grammar_get_stacks(grammar);

    for (const auto & stack : stacks) {
        if (stack.empty()) {
            return true;
        }
    }

    return false;
}

bool llama_rs_gbnf_is_rejected(struct llama_grammar * grammar) noexcept {
    return llama_grammar_get_stacks(grammar).empty();
}

llama_rs_gbnf_runtime_status llama_rs_gbnf_clone(
    const struct llama_grammar * grammar,
    struct llama_grammar ** out_clone) {
    try {
        const fault_region region;

        *out_clone = llama_grammar_clone_impl(*grammar);

        return LLAMA_RS_GBNF_RUNTIME_OK;
    } catch (...) {
        return llama_rs_capture_exception(
            nullptr,
            LLAMA_RS_GBNF_RUNTIME_ALLOCATION_FAILED,
            LLAMA_RS_GBNF_RUNTIME_THREW_CXX_EXCEPTION);
    }
}

void llama_rs_gbnf_free(struct llama_grammar * grammar) noexcept {
    llama_grammar_free_impl(grammar);
}

}
