#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum llama_rs_gbnf_validation_status {
    LLAMA_RS_GBNF_VALIDATION_OK = 0,
    LLAMA_RS_GBNF_VALIDATION_SYNTAX_ERROR,
    LLAMA_RS_GBNF_VALIDATION_EMPTY_RULE_SET,
    LLAMA_RS_GBNF_VALIDATION_ROOT_SYMBOL_MISSING,
    LLAMA_RS_GBNF_VALIDATION_LEFT_RECURSION,
    LLAMA_RS_GBNF_VALIDATION_NULL_GRAMMAR_ARG,
    LLAMA_RS_GBNF_VALIDATION_NULL_ROOT_ARG,
    LLAMA_RS_GBNF_VALIDATION_NULL_OUT_ERROR_ARG,
    LLAMA_RS_GBNF_VALIDATION_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_GBNF_VALIDATION_THREW_CXX_EXCEPTION,
} llama_rs_gbnf_validation_status;

llama_rs_gbnf_validation_status llama_rs_validate_gbnf(
    const char * grammar_str,
    const char * grammar_root,
    char ** out_error);

#ifdef __cplusplus
}
#endif
