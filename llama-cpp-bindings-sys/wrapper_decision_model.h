#pragma once

#include "wrapper_batch.h"

#include <llama.h>
#include <tools/mtmd/mtmd.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct llama_rs_decision_model;
typedef struct llama_rs_decision_model * llama_rs_decision_model_handle;

struct llama_rs_decision_request;
typedef struct llama_rs_decision_request * llama_rs_decision_request_handle;

struct llama_rs_decision_prompt;
typedef struct llama_rs_decision_prompt * llama_rs_decision_prompt_handle;

typedef enum llama_rs_decision_model_init_status {
    LLAMA_RS_DECISION_MODEL_INIT_OK = 0,
    LLAMA_RS_DECISION_MODEL_INIT_MODEL_IS_NOT_A_DECISION_MODEL,
    LLAMA_RS_DECISION_MODEL_INIT_DECISION_TYPE_UNSUPPORTED_BY_LLAMA_CPP,
    LLAMA_RS_DECISION_MODEL_INIT_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_REJECTED_MODEL,
} llama_rs_decision_model_init_status;

llama_rs_decision_model_init_status llama_rs_decision_model_init(
    const struct llama_model * model,
    llama_rs_decision_model_handle * out_model,
    char ** out_error);

void llama_rs_decision_model_free(llama_rs_decision_model_handle model);

bool llama_rs_decision_model_can_share_prompt(llama_rs_decision_model_handle model);

bool llama_rs_decision_model_is_joint(llama_rs_decision_model_handle model);

bool llama_rs_decision_model_can_use_images(llama_rs_decision_model_handle model);

typedef enum llama_rs_decision_model_output {
    LLAMA_RS_DECISION_MODEL_OUTPUT_LOGITS = 0,
    LLAMA_RS_DECISION_MODEL_OUTPUT_TOKEN_EMBEDDINGS,
    LLAMA_RS_DECISION_MODEL_OUTPUT_POOLED_EMBEDDINGS,
} llama_rs_decision_model_output;

llama_rs_decision_model_output llama_rs_decision_model_output_of(llama_rs_decision_model_handle model);

typedef enum llama_rs_decision_request_parse_status {
    LLAMA_RS_DECISION_REQUEST_PARSE_OK = 0,
    LLAMA_RS_DECISION_REQUEST_PARSE_BODY_IS_NOT_JSON,
    LLAMA_RS_DECISION_REQUEST_PARSE_REQUEST_REJECTED,
    LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_decision_request_parse_status;

llama_rs_decision_request_parse_status llama_rs_decision_request_parse(
    llama_rs_decision_model_handle model,
    const char * body_json,
    size_t body_json_length,
    llama_rs_decision_request_handle * out_request,
    char ** out_error);

void llama_rs_decision_request_free(llama_rs_decision_request_handle request);

size_t llama_rs_decision_request_question_count(llama_rs_decision_request_handle request);

size_t llama_rs_decision_request_question_variant_count(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    size_t question_index);

bool llama_rs_decision_request_has_media(llama_rs_decision_request_handle request);

typedef enum llama_rs_decision_prompt_fill_status {
    LLAMA_RS_DECISION_PROMPT_FILL_OK = 0,
    LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_UNSUPPORTED_BY_DECISION_MODEL,
    LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_REQUIRES_A_PROJECTOR,
    LLAMA_RS_DECISION_PROMPT_FILL_REQUEST_REJECTED,
    LLAMA_RS_DECISION_PROMPT_FILL_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_decision_prompt_fill_status;

llama_rs_decision_prompt_fill_status llama_rs_decision_prompt_fill(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    size_t question_index,
    size_t variant,
    mtmd_context * projector,
    llama_rs_decision_prompt_handle * out_prompt,
    char ** out_error);

llama_rs_decision_prompt_fill_status llama_rs_decision_prompt_fill_joint(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    mtmd_context * projector,
    llama_rs_decision_prompt_handle * out_prompt,
    char ** out_error);

void llama_rs_decision_prompt_free(llama_rs_decision_prompt_handle prompt);

typedef enum llama_rs_decision_prompts_share_status {
    LLAMA_RS_DECISION_PROMPTS_SHARE_OK = 0,
    LLAMA_RS_DECISION_PROMPTS_SHARE_LLAMA_CPP_OUT_OF_MEMORY,
} llama_rs_decision_prompts_share_status;

llama_rs_decision_prompts_share_status llama_rs_decision_prompts_shared_prefix(
    llama_rs_decision_prompt_handle * prompts,
    size_t prompt_count,
    size_t * out_shared_prefix_length);

const char * llama_rs_decision_media_marker(void);

const mtmd_input_chunk * llama_rs_decision_prompt_media_chunk(
    llama_rs_decision_prompt_handle prompt,
    size_t position);

typedef struct llama_rs_decision_prompt_layout {
    const llama_token * tokens;
    size_t tokens_count;
    const llama_rs_decision_order * orders;
    size_t orders_count;
    size_t first_read_position;
    bool outputs_every_token;
} llama_rs_decision_prompt_layout;

llama_rs_decision_prompt_layout llama_rs_decision_prompt_layout_of(llama_rs_decision_prompt_handle prompt);

typedef enum llama_rs_decision_answers_read_status {
    LLAMA_RS_DECISION_ANSWERS_READ_OK = 0,
    LLAMA_RS_DECISION_ANSWERS_READ_OUTPUTS_UNREADABLE,
    LLAMA_RS_DECISION_ANSWERS_READ_SCORES_REJECTED,
    LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED,
    LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_OUT_OF_MEMORY,
    LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
} llama_rs_decision_answers_read_status;

llama_rs_decision_answers_read_status llama_rs_decision_request_read_answers(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    llama_rs_decision_prompt_handle prompt,
    struct llama_context * context,
    const int32_t * batch_indices,
    size_t batch_indices_count,
    llama_seq_id sequence_id,
    char ** out_answers_json,
    char ** out_error);

#ifdef __cplusplus
}
#endif
