#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#define LLAMA_RS_GBNF_NOEXCEPT noexcept
#else
#define LLAMA_RS_GBNF_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct llama_grammar;

typedef enum llama_rs_gbnf_parse_status {
    LLAMA_RS_GBNF_PARSE_OK = 0,
    LLAMA_RS_GBNF_PARSE_SYNTAX_ERROR,
    LLAMA_RS_GBNF_PARSE_EMPTY_RULE_SET,
    LLAMA_RS_GBNF_PARSE_ROOT_SYMBOL_MISSING,
    LLAMA_RS_GBNF_PARSE_LEFT_RECURSION,
    LLAMA_RS_GBNF_PARSE_ALLOCATION_FAILED,
    LLAMA_RS_GBNF_PARSE_THREW_CXX_EXCEPTION,
} llama_rs_gbnf_parse_status;

typedef enum llama_rs_gbnf_runtime_status {
    LLAMA_RS_GBNF_RUNTIME_OK = 0,
    LLAMA_RS_GBNF_RUNTIME_ALLOCATION_FAILED,
    LLAMA_RS_GBNF_RUNTIME_THREW_CXX_EXCEPTION,
} llama_rs_gbnf_runtime_status;

llama_rs_gbnf_parse_status llama_rs_gbnf_parse(
    const char * grammar_str,
    const char * grammar_root,
    struct llama_grammar ** out_grammar);

llama_rs_gbnf_runtime_status llama_rs_gbnf_accept_str(
    struct llama_grammar * grammar,
    const char * piece,
    size_t piece_len);

bool llama_rs_gbnf_is_accepting(struct llama_grammar * grammar) LLAMA_RS_GBNF_NOEXCEPT;

bool llama_rs_gbnf_is_rejected(struct llama_grammar * grammar) LLAMA_RS_GBNF_NOEXCEPT;

llama_rs_gbnf_runtime_status llama_rs_gbnf_clone(
    const struct llama_grammar * grammar,
    struct llama_grammar ** out_clone);

void llama_rs_gbnf_free(struct llama_grammar * grammar) LLAMA_RS_GBNF_NOEXCEPT;

#ifdef __cplusplus
}
#endif
