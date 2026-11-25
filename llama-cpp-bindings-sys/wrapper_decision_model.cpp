#include "wrapper_decision_model.h"
#include "wrapper_batch.h"
#include "wrapper_fault_injection.h"
#include "wrapper_report_current_exception.h"
#include "wrapper_utils.h"

#include <common/common.h>
#include <common/json.h>
#include <llama.h>
#include <server-common.h>
#include <server-decision.h>
#include <server-task.h>
#include <tools/mtmd/mtmd-helper.h>
#include <tools/mtmd/mtmd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gsl/span>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct llama_rs_decision_model {
    server_decision_context context;
};

struct llama_rs_decision_request {
    json state;
    std::vector<raw_buffer> files;
    std::vector<server_decision_question> questions;
    std::vector<std::vector<std::vector<float>>> variant_scores;
};

struct llama_rs_decision_prompt {
    server_task task = server_task(SERVER_TASK_TYPE_DECISION);
    std::vector<llama_token> token_ids;
    std::vector<llama_rs_decision_order> orders;
    bool joint = false;
    size_t question_index = 0;
};

namespace {
using server_decision_layout = decltype(server_task::decision);

auto decision_order_of(int32_t order) -> llama_rs_decision_order {
    switch (order) {
        case LLAMA_RS_DECISION_ORDER_NONE:
            return LLAMA_RS_DECISION_ORDER_NONE;
        case LLAMA_RS_DECISION_ORDER_QUESTION_NOUL:
            return LLAMA_RS_DECISION_ORDER_QUESTION_NOUL;
        case LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE:
            return LLAMA_RS_DECISION_ORDER_QUESTION_CHOICE;
        case LLAMA_RS_DECISION_ORDER_QUESTION_SCORE:
            return LLAMA_RS_DECISION_ORDER_QUESTION_SCORE;
        case LLAMA_RS_DECISION_ORDER_OPTION:
            return LLAMA_RS_DECISION_ORDER_OPTION;
        default:
            throw std::logic_error("llama.cpp laid out a decision order the binding does not know");
    }
}

class decision_outputs_unreadable : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

auto embedding_scores(
    const server_decision_layout & decision,
    llama_context * context,
    gsl::span<const int32_t> batch_indices,
    int32_t prompt_tokens) -> std::vector<float> {
    const auto outputs = static_cast<int32_t>(batch_indices.size());
    const int32_t pos_first = prompt_tokens - outputs;
    const int32_t n_embd_out = llama_model_n_embd_out(llama_get_model(context));
    const auto embeddings_of_output = [&](int32_t output) -> gsl::span<const float> {
        const float * embeddings =
            output >= 0 && output < outputs ? llama_get_embeddings_ith(context, batch_indices[output]) : nullptr;

        return embeddings == nullptr ? gsl::span<const float>()
                                     : gsl::span<const float>(embeddings, static_cast<size_t>(n_embd_out));
    };
    std::vector<float> scores;

    for (int32_t row = 0; row < decision.n_scores; row++) {
        const gsl::span<const float> embeddings = embeddings_of_output(row);

        if (embeddings.empty()) {
            throw decision_outputs_unreadable("failed to get embeddings");
        }
        scores.push_back(embeddings.front());
    }

    const int32_t n_pointer = n_embd_out / 2;
    const gsl::span<const float> query =
        decision.pointer >= 0 ? embeddings_of_output(decision.pointer - pos_first) : gsl::span<const float>();

    if (decision.column < 0 || decision.column >= n_embd_out) {
        throw decision_outputs_unreadable("the decision column lies outside the embeddings");
    }

    for (const int32_t marker : decision.markers) {
        const gsl::span<const float> marked = embeddings_of_output(marker - pos_first);

        if (marked.empty() || (decision.pointer >= 0 && query.empty())) {
            throw decision_outputs_unreadable(
                "failed to get embeddings, the question and its options must fit in one batch");
        }
        if (decision.pointer < 0) {
            scores.push_back(marked[static_cast<size_t>(decision.column)]);
            continue;
        }

        const gsl::span<const float> query_head = query.first(static_cast<size_t>(n_pointer));
        const gsl::span<const float> marked_head = marked.subspan(static_cast<size_t>(n_pointer), query_head.size());

        scores.push_back(std::inner_product(query_head.begin(), query_head.end(), marked_head.begin(), 0.0F) /
                         sqrtf(static_cast<float>(n_pointer)));
    }

    return scores;
}

auto label_scores(
    const server_decision_layout & decision,
    llama_context * context,
    gsl::span<const int32_t> batch_indices) -> std::vector<float> {
    if (batch_indices.empty()) {
        throw decision_outputs_unreadable("failed to get logits");
    }

    const float * logits = llama_get_logits_ith(context, batch_indices.back());

    if (logits == nullptr) {
        throw decision_outputs_unreadable("failed to get logits");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(context)));
    const gsl::span<const float> vocabulary_logits(logits, static_cast<size_t>(n_vocab));
    std::vector<float> scores;

    for (const llama_token label : decision.labels) {
        if (label < 0 || label >= n_vocab) {
            throw decision_outputs_unreadable("a decision label lies outside the vocabulary");
        }
        scores.push_back(vocabulary_logits[static_cast<size_t>(label)]);
    }

    if (decision.label_groups.empty()) {
        return scores;
    }

    std::vector<float> grouped;
    size_t first = 0;

    for (const int32_t group_size : decision.label_groups) {
        if (group_size <= 0 || first + static_cast<size_t>(group_size) > scores.size()) {
            throw decision_outputs_unreadable("a decision label group lies outside the labels");
        }

        const auto group_begin = scores.begin() + static_cast<std::ptrdiff_t>(first);

        grouped.push_back(*std::max_element(group_begin, group_begin + group_size));
        first += static_cast<size_t>(group_size);
    }

    return grouped;
}

auto prompt_scores(
    const llama_rs_decision_prompt & prompt,
    llama_context * context,
    gsl::span<const int32_t> batch_indices,
    llama_seq_id sequence_id) -> std::vector<float> {
    const auto & decision = prompt.task.decision;

    if (decision.pooled) {
        const float * pooled = llama_get_embeddings_seq(context, sequence_id);

        if (pooled == nullptr) {
            throw decision_outputs_unreadable("failed to get the class scores");
        }

        const gsl::span<const float> class_scores(
            pooled, static_cast<size_t>(llama_model_n_embd_out(llama_get_model(context))));

        return {class_scores.begin(), class_scores.end()};
    }
    if (!decision.labels.empty()) {
        return label_scores(decision, context, batch_indices);
    }

    return embedding_scores(decision, context, batch_indices, prompt.task.n_tokens());
}
}  // namespace

namespace {
template <typename TFill>
auto fill_prompt(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    mtmd_context * projector,
    llama_rs_decision_prompt_handle * out_prompt,
    char ** out_error,
    bool joint,
    size_t question_index,
    TFill fill) -> llama_rs_decision_prompt_fill_status {
    *out_prompt = nullptr;
    *out_error = nullptr;

    try {
        if (!request->files.empty() && !model->context.can_use_images()) {
            return LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_UNSUPPORTED_BY_DECISION_MODEL;
        }
        if (!request->files.empty() && projector == nullptr) {
            return LLAMA_RS_DECISION_PROMPT_FILL_MEDIA_REQUIRES_A_PROJECTOR;
        }

        auto prompt = std::make_unique<llama_rs_decision_prompt>();

        prompt->joint = joint;
        prompt->question_index = question_index;

        try {
            fill(mtmd_helper_init_opt_default(), prompt->task);
        } catch (const std::invalid_argument & err) {
            return llama_rs_report_exception_message(
                out_error,
                err.what(),
                LLAMA_RS_DECISION_PROMPT_FILL_REQUEST_REJECTED,
                LLAMA_RS_DECISION_PROMPT_FILL_ERROR_STRING_ALLOCATION_FAILED);
        }

        const auto & tokens = prompt->task.tokens;

        prompt->token_ids.reserve(tokens.size());
        for (size_t position = 0; position < tokens.size(); ++position) {
            prompt->token_ids.push_back(tokens[position]);
        }

        prompt->orders.reserve(prompt->task.decision.order.size());
        for (const int32_t order : prompt->task.decision.order) {
            prompt->orders.push_back(decision_order_of(order));
        }

        *out_prompt = prompt.release();

        return LLAMA_RS_DECISION_PROMPT_FILL_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_DECISION_PROMPT_FILL_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_DECISION_PROMPT_FILL_ERROR_STRING_ALLOCATION_FAILED);
    }
}
}  // namespace

extern "C" auto llama_rs_decision_model_init(
    const struct llama_model * model,
    llama_rs_decision_model_handle * out_model,
    char ** out_error) -> llama_rs_decision_model_init_status {
    *out_model = nullptr;
    *out_error = nullptr;

    try {
        const common_decision_type decision_type = common_get_decision_type(model);

        if (decision_type == COMMON_DECISION_TYPE_NONE) {
            return LLAMA_RS_DECISION_MODEL_INIT_MODEL_IS_NOT_A_DECISION_MODEL;
        }
        if (decision_type == COMMON_DECISION_TYPE_UNKNOWN) {
            return LLAMA_RS_DECISION_MODEL_INIT_DECISION_TYPE_UNSUPPORTED_BY_LLAMA_CPP;
        }

        auto decision_model = std::make_unique<llama_rs_decision_model>();

        decision_model->context.init(model);
        *out_model = decision_model.release();

        return LLAMA_RS_DECISION_MODEL_INIT_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_DECISION_MODEL_INIT_LLAMA_CPP_REJECTED_MODEL,
            LLAMA_RS_DECISION_MODEL_INIT_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_decision_model_free(llama_rs_decision_model_handle model) -> void {
    std::unique_ptr<llama_rs_decision_model>(model).reset();
}

extern "C" auto llama_rs_decision_model_can_share_prompt(llama_rs_decision_model_handle model) -> bool {
    return model->context.can_share_prompt();
}

extern "C" auto llama_rs_decision_model_is_joint(llama_rs_decision_model_handle model) -> bool {
    return model->context.is_joint();
}

extern "C" auto llama_rs_decision_model_can_use_images(llama_rs_decision_model_handle model) -> bool {
    return model->context.can_use_images();
}

extern "C" auto llama_rs_decision_model_output_of(llama_rs_decision_model_handle model)
    -> llama_rs_decision_model_output {
    switch (model->context.type) {
        case COMMON_DECISION_TYPE_LAYA:
        case COMMON_DECISION_TYPE_KEV:
        case COMMON_DECISION_TYPE_CLEF:
        case COMMON_DECISION_TYPE_LFM2_D1_OMNI:
        case COMMON_DECISION_TYPE_VLLM_DECISION2:
            return LLAMA_RS_DECISION_MODEL_OUTPUT_TOKEN_EMBEDDINGS;
        case COMMON_DECISION_TYPE_MINICPM_OCR_JEV:
            return LLAMA_RS_DECISION_MODEL_OUTPUT_POOLED_EMBEDDINGS;
        default:
            return LLAMA_RS_DECISION_MODEL_OUTPUT_LOGITS;
    }
}

extern "C" auto llama_rs_decision_request_parse(
    llama_rs_decision_model_handle model,
    const char * body_json,
    size_t body_json_length,
    llama_rs_decision_request_handle * out_request,
    char ** out_error) -> llama_rs_decision_request_parse_status {
    *out_request = nullptr;
    *out_error = nullptr;

    try {
        json body;

        try {
            body = json::parse(std::string(body_json, body_json_length));
        } catch (const common_json_error & err) {
            return llama_rs_report_exception_message(
                out_error,
                err.what(),
                LLAMA_RS_DECISION_REQUEST_PARSE_BODY_IS_NOT_JSON,
                LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED);
        }

        auto request = std::make_unique<llama_rs_decision_request>();

        try {
            request->questions = model->context.parse_questions(body);
            request->state = model->context.prepare_state(model->context.parse_state(body, request->files));
            request->variant_scores.resize(request->questions.size());
        } catch (const std::invalid_argument & err) {
            return llama_rs_report_exception_message(
                out_error,
                err.what(),
                LLAMA_RS_DECISION_REQUEST_PARSE_REQUEST_REJECTED,
                LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED);
        }

        *out_request = request.release();

        return LLAMA_RS_DECISION_REQUEST_PARSE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_DECISION_REQUEST_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_DECISION_REQUEST_PARSE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_decision_request_free(llama_rs_decision_request_handle request) -> void {
    std::unique_ptr<llama_rs_decision_request>(request).reset();
}

extern "C" auto llama_rs_decision_request_question_count(llama_rs_decision_request_handle request) -> size_t {
    return request->questions.size();
}

extern "C" auto llama_rs_decision_request_question_variant_count(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    size_t question_index) -> size_t {
    return model->context.n_variants(request->questions.at(question_index));
}

extern "C" auto llama_rs_decision_request_has_media(llama_rs_decision_request_handle request) -> bool {
    return !request->files.empty();
}

extern "C" auto llama_rs_decision_prompt_fill(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    size_t question_index,
    size_t variant,
    mtmd_context * projector,
    llama_rs_decision_prompt_handle * out_prompt,
    char ** out_error) -> llama_rs_decision_prompt_fill_status {
    return fill_prompt(
        model,
        request,
        projector,
        out_prompt,
        out_error,
        false,
        question_index,
        [&](const mtmd_helper_init_opt & init_opt, server_task & task) -> void {
            model->context.fill_task(
                request->state,
                request->questions,
                request->questions.at(question_index),
                variant,
                request->files,
                projector,
                init_opt,
                task);
        });
}

extern "C" auto llama_rs_decision_prompt_fill_joint(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    mtmd_context * projector,
    llama_rs_decision_prompt_handle * out_prompt,
    char ** out_error) -> llama_rs_decision_prompt_fill_status {
    return fill_prompt(
        model,
        request,
        projector,
        out_prompt,
        out_error,
        true,
        0,
        [&](const mtmd_helper_init_opt & init_opt, server_task & task) -> void {
            model->context.fill_task_joint(
                request->state, request->questions, request->files, projector, init_opt, task);
        });
}

extern "C" auto llama_rs_decision_prompt_free(llama_rs_decision_prompt_handle prompt) -> void {
    std::unique_ptr<llama_rs_decision_prompt>(prompt).reset();
}

extern "C" auto llama_rs_decision_prompts_shared_prefix(
    llama_rs_decision_prompt_handle * prompts,
    size_t prompt_count,
    size_t * out_shared_prefix_length) -> llama_rs_decision_prompts_share_status {
    const gsl::span<llama_rs_decision_prompt_handle> handles(prompts, prompt_count);

    try {
        if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_DECISION_PROMPTS_SHARE_RUNS_OUT_OF_MEMORY)) {
            throw std::bad_alloc();
        }

        std::vector<server_task> tasks;

        tasks.reserve(prompt_count);
        std::transform(handles.begin(), handles.end(), std::back_inserter(tasks),
            [](llama_rs_decision_prompt * prompt) -> server_task { return std::move(prompt->task); });

        std::vector<server_task> groups = server_decision_group_tasks(std::move(tasks), prompt_count);

        if (groups.size() == prompt_count) {
            for (size_t prompt_index = 0; prompt_index < prompt_count; prompt_index++) {
                handles[prompt_index]->task = std::move(groups[prompt_index]);
            }

            *out_shared_prefix_length = 0;

            return LLAMA_RS_DECISION_PROMPTS_SHARE_OK;
        }

        server_task & parent = groups.front();

        for (size_t prompt_index = 1; prompt_index < prompt_count; prompt_index++) {
            handles[prompt_index]->task = std::move(parent.child_tasks[prompt_index - 1]);
        }

        parent.child_tasks.clear();
        *out_shared_prefix_length = static_cast<size_t>(parent.n_tokens_shared);
        handles[0]->task = std::move(parent);

        return LLAMA_RS_DECISION_PROMPTS_SHARE_OK;
    } catch (const std::bad_alloc &) {
        return LLAMA_RS_DECISION_PROMPTS_SHARE_LLAMA_CPP_OUT_OF_MEMORY;
    }
}

extern "C" auto llama_rs_decision_media_marker() -> const char * {
    return get_media_marker();
}

extern "C" auto llama_rs_decision_prompt_media_chunk(llama_rs_decision_prompt_handle prompt, size_t position)
    -> const mtmd_input_chunk * {
    const auto & tokens = prompt->task.tokens;

    if (position >= tokens.size() || tokens[position] != LLAMA_TOKEN_NULL) {
        return nullptr;
    }

    try {
        return tokens.find_chunk(position).get();
    } catch (const std::runtime_error &) {
        return nullptr;
    }
}

extern "C" auto llama_rs_decision_prompt_layout_of(llama_rs_decision_prompt_handle prompt)
    -> llama_rs_decision_prompt_layout {
    const int32_t pos_first = prompt->task.decision.pos_first();
    const size_t last_position = prompt->token_ids.empty() ? 0 : prompt->token_ids.size() - 1;

    return {
        prompt->token_ids.data(),
        prompt->token_ids.size(),
        prompt->orders.data(),
        prompt->orders.size(),
        pos_first < 0 ? last_position : static_cast<size_t>(pos_first),
        prompt->task.decision.labels.empty() && !prompt->task.decision.pooled,
    };
}

extern "C" auto llama_rs_decision_request_read_answers(
    llama_rs_decision_model_handle model,
    llama_rs_decision_request_handle request,
    llama_rs_decision_prompt_handle prompt,
    struct llama_context * context,
    const int32_t * batch_indices,
    size_t batch_indices_count,
    llama_seq_id sequence_id,
    char ** out_answers_json,
    char ** out_error) -> llama_rs_decision_answers_read_status {
    *out_answers_json = nullptr;
    *out_error = nullptr;

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_DECISION_ANSWERS_READ_THROWS);

        std::vector<float> scores;

        try {
            scores = prompt_scores(
                *prompt, context, gsl::span<const int32_t>(batch_indices, batch_indices_count), sequence_id);
        } catch (const decision_outputs_unreadable & err) {
            return llama_rs_report_exception_message(
                out_error,
                err.what(),
                LLAMA_RS_DECISION_ANSWERS_READ_OUTPUTS_UNREADABLE,
                LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED);
        }

        std::vector<size_t> answered_questions;

        if (prompt->joint) {
            size_t first = 0;

            for (size_t question_index = 0; question_index < request->questions.size(); ++question_index) {
                const size_t option_count = request->questions[question_index].options.size();

                if (first + option_count > scores.size()) {
                    return llama_rs_report_exception_message(
                        out_error,
                        "the joint scores do not cover every question",
                        LLAMA_RS_DECISION_ANSWERS_READ_OUTPUTS_UNREADABLE,
                        LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED);
                }

                const auto question_begin = scores.begin() + static_cast<std::ptrdiff_t>(first);

                request->variant_scores[question_index].emplace_back(question_begin, question_begin + static_cast<std::ptrdiff_t>(option_count));
                answered_questions.push_back(question_index);
                first += option_count;
            }
        } else {
            auto & variant_scores = request->variant_scores.at(prompt->question_index);

            variant_scores.push_back(std::move(scores));
            if (variant_scores.size() == model->context.n_variants(request->questions[prompt->question_index])) {
                answered_questions.push_back(prompt->question_index);
            }
        }

        json answers = json::object();

        for (const size_t question_index : answered_questions) {
            const auto & question = request->questions[question_index];

            try {
                answers[question.id] = model->context.format_answer(
                    question, request->variant_scores[question_index], !request->files.empty());
            } catch (const std::runtime_error & err) {
                return llama_rs_report_exception_message(
                    out_error,
                    err.what(),
                    LLAMA_RS_DECISION_ANSWERS_READ_SCORES_REJECTED,
                    LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED);
            }
        }

        *out_answers_json = llama_rs_dup_string(answers.dump());
        if (*out_answers_json == nullptr) {
            return LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_DECISION_ANSWERS_READ_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_DECISION_ANSWERS_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_DECISION_ANSWERS_READ_ERROR_STRING_ALLOCATION_FAILED);
    }
}
