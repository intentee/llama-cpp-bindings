#include "wrapper_gbnf.h"
#include "wrapper_report_current_exception.h"

#include <src/llama-grammar.h>

extern "C" auto llama_rs_validate_gbnf(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root,
    char ** out_error) -> llama_rs_gbnf_validation_status {
    *out_error = nullptr;
    try {
        llama_grammar_parser parser(vocab);

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
            vocab, grammar_str, grammar_root, false, nullptr, 0, nullptr, 0);

        if (grammar == nullptr) {
            return LLAMA_RS_GBNF_VALIDATION_UNSUPPORTED_GRAMMAR;
        }

        llama_grammar_free_impl(grammar);

        return LLAMA_RS_GBNF_VALIDATION_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_GBNF_VALIDATION_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
            LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED);
    }
}
