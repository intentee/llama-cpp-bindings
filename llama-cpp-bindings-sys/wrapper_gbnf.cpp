#include "wrapper_gbnf.h"

#include "llama.cpp/src/llama-grammar.h"

extern "C" auto llama_rs_validate_gbnf(
    const char * grammar_str,
    const char * grammar_root) -> llama_rs_gbnf_validation_status {
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
    } catch (...) {
        return LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION;
    }
}
