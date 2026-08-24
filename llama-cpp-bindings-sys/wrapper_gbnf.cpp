#include "wrapper_gbnf.h"
#include "wrapper_utils.h"

#include "llama.cpp/src/llama-grammar.h"

#include <exception>
#include <new>

extern "C" auto llama_rs_validate_gbnf(
    const char * grammar_str,
    const char * grammar_root,
    char ** out_error) -> llama_rs_gbnf_validation_status {
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (grammar_str == nullptr) {
        return LLAMA_RS_GBNF_VALIDATION_NULL_GRAMMAR_ARG;
    }
    if (grammar_root == nullptr) {
        return LLAMA_RS_GBNF_VALIDATION_NULL_ROOT_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_GBNF_VALIDATION_NULL_OUT_ERROR_ARG;
    }
    try {
        llama_grammar_parser parser;

        if (!parser.parse(grammar_str)) {
            return LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR;
        }

        if (parser.rules.empty()) {
            return LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET;
        }

        if (parser.symbol_ids.find(grammar_root) == parser.symbol_ids.end()) {
            return LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING;
        }

        llama_grammar * grammar = llama_grammar_init_impl(
            nullptr, grammar_str, grammar_root, false, nullptr, 0, nullptr, 0);

        if (grammar == nullptr) {
            return LLAMA_RS_GBNF_VALIDATION_LEFT_RECURSION;
        }

        llama_grammar_free_impl(grammar);

        return LLAMA_RS_GBNF_VALIDATION_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED;
    } catch (const std::exception & err) {
        *out_error = llama_rs_dup_string(err.what());
        if (*out_error == nullptr) {
            return LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
    } catch (...) {
        *out_error = llama_rs_dup_string("unknown c++ exception");
        if (*out_error == nullptr) {
            return LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED;
        }
        return LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
    }
}
