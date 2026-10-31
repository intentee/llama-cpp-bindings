#pragma once

#include <llama.h>
#include "wrapper_parsed_chat.h"
#include "wrapper_chat_templates.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_chat_session;
typedef struct llama_rs_chat_session * llama_rs_chat_session_handle;

struct llama_rs_chat_spans;
typedef struct llama_rs_chat_spans * llama_rs_chat_spans_handle;

typedef enum llama_rs_chat_tool_choice {
    LLAMA_RS_CHAT_TOOL_CHOICE_AUTO = 0,
    LLAMA_RS_CHAT_TOOL_CHOICE_NONE,
    LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED,
} llama_rs_chat_tool_choice;

typedef enum llama_rs_chat_assistant_turn {
    LLAMA_RS_CHAT_ASSISTANT_TURN_OMITTED = 0,
    LLAMA_RS_CHAT_ASSISTANT_TURN_NEW,
    LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_REASONING,
    LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_CONTENT,
} llama_rs_chat_assistant_turn;

typedef struct llama_rs_chat_session_options {
    llama_rs_chat_assistant_turn assistant_turn;
    bool enable_thinking;
    bool parallel_tool_calls;
    llama_rs_chat_tool_choice tool_choice;
} llama_rs_chat_session_options;

typedef enum llama_rs_chat_session_create_status {
    LLAMA_RS_CHAT_SESSION_CREATE_OK = 0,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_TEMPLATES_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_VOCAB_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_MESSAGES_JSON_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_TOOLS_JSON_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_SESSION_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_NOT_JSON,
    LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_INVALID,
    LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_NOT_JSON,
    LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_INVALID,
    LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
    LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_chat_session_create_status;

llama_rs_chat_session_create_status llama_rs_chat_session_create(
    llama_rs_chat_templates_handle templates,
    const struct llama_vocab * vocab,
    const char * messages_json,
    const char * tools_json,
    llama_rs_chat_session_options options,
    llama_rs_chat_session_handle * out_session,
    char ** out_error);

void llama_rs_chat_session_free(llama_rs_chat_session_handle session);

size_t llama_rs_chat_session_prompt_part_count(llama_rs_chat_session_handle session);

const char * llama_rs_chat_session_prompt_part_text(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length);

bool llama_rs_chat_session_prompt_part_is_input(llama_rs_chat_session_handle session, size_t index);

const char * llama_rs_chat_session_grammar(llama_rs_chat_session_handle session, size_t * out_length);

bool llama_rs_chat_session_grammar_lazy(llama_rs_chat_session_handle session);

typedef enum llama_rs_chat_grammar_trigger_type {
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN = 0,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
    LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED,
} llama_rs_chat_grammar_trigger_type;

size_t llama_rs_chat_session_grammar_trigger_count(llama_rs_chat_session_handle session);

llama_rs_chat_grammar_trigger_type llama_rs_chat_session_grammar_trigger_type(
    llama_rs_chat_session_handle session,
    size_t index);

const char * llama_rs_chat_session_grammar_trigger_value(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length);

llama_token llama_rs_chat_session_grammar_trigger_token(llama_rs_chat_session_handle session, size_t index);

size_t llama_rs_chat_session_preserved_token_count(llama_rs_chat_session_handle session);

llama_token llama_rs_chat_session_preserved_token(llama_rs_chat_session_handle session, size_t index);

const char * llama_rs_chat_session_generation_prompt(llama_rs_chat_session_handle session, size_t * out_length);

size_t llama_rs_chat_session_generation_prompt_token_count(llama_rs_chat_session_handle session);

llama_token llama_rs_chat_session_generation_prompt_token(llama_rs_chat_session_handle session, size_t index);

const char * llama_rs_chat_session_thinking_start_tag(llama_rs_chat_session_handle session, size_t * out_length);

size_t llama_rs_chat_session_thinking_end_tag_count(llama_rs_chat_session_handle session);

const char * llama_rs_chat_session_thinking_end_tag(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length);

typedef enum llama_rs_chat_session_parse_status {
    LLAMA_RS_CHAT_SESSION_PARSE_OK = 0,
    LLAMA_RS_CHAT_SESSION_PARSE_NULL_SESSION_ARG,
    LLAMA_RS_CHAT_SESSION_PARSE_NULL_TEXT_ARG,
    LLAMA_RS_CHAT_SESSION_PARSE_NULL_BYTE_TOKENS_ARG,
    LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_HANDLE_ARG,
    LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_CHAT_SESSION_PARSE_OUTPUT_DOES_NOT_MATCH_FORMAT,
    LLAMA_RS_CHAT_SESSION_PARSE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_chat_session_parse_status;

llama_rs_chat_session_parse_status llama_rs_chat_session_parse(
    llama_rs_chat_session_handle session,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens,
    bool is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error);

typedef enum llama_rs_chat_span_kind {
    LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK = 0,
    LLAMA_RS_CHAT_SPAN_KIND_REASONING,
    LLAMA_RS_CHAT_SPAN_KIND_CONTENT,
    LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL,
    LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER,
} llama_rs_chat_span_kind;

typedef struct llama_rs_chat_span {
    llama_rs_chat_span_kind kind;
    size_t start;
    size_t end;
} llama_rs_chat_span;

typedef struct llama_rs_chat_spans_view {
    const llama_rs_chat_span * spans;
    size_t spans_count;
    size_t parsed_length;
} llama_rs_chat_spans_view;

typedef enum llama_rs_chat_session_classify_status {
    LLAMA_RS_CHAT_SESSION_CLASSIFY_OK = 0,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_SESSION_ARG,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_TEXT_ARG,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_BYTE_TOKENS_ARG,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_SPANS_ARG,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_ERROR_ARG,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_OUTPUT_DOES_NOT_MATCH_FORMAT,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_chat_session_classify_status;

llama_rs_chat_session_classify_status llama_rs_chat_session_classify(
    llama_rs_chat_session_handle session,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens,
    bool is_partial,
    llama_rs_chat_spans_handle * out_spans,
    char ** out_error);

llama_rs_chat_spans_view llama_rs_chat_spans_view_of(llama_rs_chat_spans_handle spans);

void llama_rs_chat_spans_free(llama_rs_chat_spans_handle spans);

typedef enum llama_rs_regex_escape_status {
    LLAMA_RS_REGEX_ESCAPE_OK = 0,
    LLAMA_RS_REGEX_ESCAPE_NULL_TEXT_ARG,
    LLAMA_RS_REGEX_ESCAPE_NULL_OUT_STRING_ARG,
    LLAMA_RS_REGEX_ESCAPE_NULL_OUT_ERROR_ARG,
    LLAMA_RS_REGEX_ESCAPE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_regex_escape_status;

llama_rs_regex_escape_status llama_rs_regex_escape(
    const char * text,
    size_t text_length,
    char ** out_string,
    char ** out_error);

#ifdef __cplusplus
}
#endif
