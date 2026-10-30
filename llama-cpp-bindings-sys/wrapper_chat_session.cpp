#include "wrapper_chat_session.h"
#include "wrapper_chat_templates.h"
#include "wrapper_chat_templates_handle.h"
#include "wrapper_parsed_chat.h"
#include "wrapper_parsed_chat_handle.h"
#include "wrapper_report_current_exception.h"
#include "wrapper_utils.h"

#include "llama.cpp/common/chat-peg-parser.h"
#include "llama.cpp/common/chat.h"
#include "llama.cpp/common/common.h"
#include "llama.cpp/common/json.h"
#include "llama.cpp/common/peg-parser.h"
#include "llama.cpp/include/llama.h"

#include <cstddef>
#include <exception>
#include <gsl/span>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

struct llama_rs_chat_session {
    common_chat_session session;
    common_chat_input generation_prompt;
    std::vector<llama_token> generation_prompt_tokens;
    common_params_sampling sampling;
    std::vector<llama_token> preserved_tokens;
};

struct llama_rs_chat_span {
    llama_rs_chat_span_kind kind;
    size_t start;
    size_t end;
};

struct llama_rs_chat_spans {
    size_t parsed_length = 0;
    std::vector<llama_rs_chat_span> spans;
};

namespace {
auto tool_choice_of(llama_rs_chat_tool_choice tool_choice) -> common_chat_tool_choice {
    switch (tool_choice) {
        case LLAMA_RS_CHAT_TOOL_CHOICE_NONE:
            return COMMON_CHAT_TOOL_CHOICE_NONE;
        case LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED:
            return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        case LLAMA_RS_CHAT_TOOL_CHOICE_AUTO:
            return COMMON_CHAT_TOOL_CHOICE_AUTO;
    }

    return COMMON_CHAT_TOOL_CHOICE_AUTO;
}

auto continuation_of(llama_rs_chat_assistant_turn assistant_turn) -> common_chat_continuation {
    switch (assistant_turn) {
        case LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_REASONING:
            return COMMON_CHAT_CONTINUATION_REASONING;
        case LLAMA_RS_CHAT_ASSISTANT_TURN_CONTINUE_CONTENT:
            return COMMON_CHAT_CONTINUATION_CONTENT;
        case LLAMA_RS_CHAT_ASSISTANT_TURN_OMITTED:
        case LLAMA_RS_CHAT_ASSISTANT_TURN_NEW:
            return COMMON_CHAT_CONTINUATION_NONE;
    }

    return COMMON_CHAT_CONTINUATION_NONE;
}

auto span_kind_of_tag(const std::string & tag) -> std::optional<llama_rs_chat_span_kind> {
    if (tag == common_chat_peg_builder::REASONING_BLOCK) {
        return LLAMA_RS_CHAT_SPAN_KIND_REASONING_BLOCK;
    }
    if (tag == common_chat_peg_builder::REASONING) {
        return LLAMA_RS_CHAT_SPAN_KIND_REASONING;
    }
    if (tag == common_chat_peg_builder::CONTENT) {
        return LLAMA_RS_CHAT_SPAN_KIND_CONTENT;
    }
    if (tag == common_chat_peg_builder::TOOL_OPEN || tag == common_chat_peg_builder::TOOL_CLOSE ||
        tag == common_chat_peg_builder::TOOL_ARG_OPEN || tag == common_chat_peg_builder::TOOL_ARG_CLOSE) {
        return LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER;
    }
    if (tag == common_chat_peg_builder::TOOL || tag == common_chat_peg_builder::TOOL_ID ||
        tag == common_chat_peg_builder::TOOL_NAME || tag == common_chat_peg_builder::TOOL_ARGS ||
        tag == common_chat_peg_builder::TOOL_ARG || tag == common_chat_peg_builder::TOOL_ARG_NAME ||
        tag == common_chat_peg_builder::TOOL_ARG_VALUE || tag == common_chat_peg_builder::TOOL_ARG_STRING_VALUE ||
        tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_OBJECT || tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ARRAY ||
        tag == common_chat_peg_minimax_m3_mapper::TOOL_ARG_ITEM) {
        return LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL;
    }

    return std::nullopt;
}

auto is_trigger_rule(const common_peg_arena & parser, const std::string & rule) -> bool {
    if (rule.empty() || !parser.has_rule(rule)) {
        return false;
    }

    const auto * rule_parser = std::get_if<common_peg_rule_parser>(&parser.get(parser.get_rule(rule)));

    return rule_parser != nullptr && rule_parser->trigger;
}

auto span_kind_of_node(const common_peg_arena & parser, const common_peg_ast_node & node)
    -> std::optional<llama_rs_chat_span_kind> {
    if (const auto kind = span_kind_of_tag(node.tag)) {
        return kind;
    }
    if (is_trigger_rule(parser, node.rule)) {
        return LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER;
    }

    return std::nullopt;
}

auto chat_input_of(const char * text, size_t text_length, const llama_token * byte_tokens) -> common_chat_input {
    common_chat_input input;
    input.text.assign(text, text_length);
    const gsl::span<const llama_token> tokens(byte_tokens, text_length);
    input.tokens.assign(tokens.begin(), tokens.end());

    return input;
}

auto session_parser(const common_chat_session & session) -> const common_peg_arena & {
    static const common_peg_arena content_only =
        build_chat_peg_parser([](common_chat_peg_builder & builder) -> common_peg_parser {
            return builder.content(builder.rest()) + builder.end();
        });

    return session.parser().empty() ? content_only : session.parser();
}
}  // namespace

extern "C" auto llama_rs_chat_session_create(
    llama_rs_chat_templates_handle templates,
    const struct llama_vocab * vocab,
    const char * messages_json,
    const char * tools_json,
    llama_rs_chat_session_options options,
    llama_rs_chat_session_handle * out_session,
    char ** out_error) -> llama_rs_chat_session_create_status {
    if (out_session != nullptr) {
        *out_session = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (templates == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_TEMPLATES_ARG;
    }
    if (vocab == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_VOCAB_ARG;
    }
    if (messages_json == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_MESSAGES_JSON_ARG;
    }
    if (tools_json == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_TOOLS_JSON_ARG;
    }
    if (out_session == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_SESSION_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CREATE_NULL_OUT_ERROR_ARG;
    }

    try {
        common_chat_templates_inputs inputs;

        try {
            inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages_json));
        } catch (const common_json_error & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_NOT_JSON,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        } catch (const std::invalid_argument & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_MESSAGES_INVALID,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        }

        try {
            inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools_json));
        } catch (const common_json_error & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_NOT_JSON,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        } catch (const std::runtime_error & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_TOOLS_INVALID,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        }

        inputs.add_generation_prompt = options.assistant_turn == LLAMA_RS_CHAT_ASSISTANT_TURN_NEW;
        inputs.continue_final_message = continuation_of(options.assistant_turn);
        inputs.enable_thinking = options.enable_thinking;
        inputs.parallel_tool_calls = options.parallel_tool_calls;
        inputs.tool_choice = tool_choice_of(options.tool_choice);
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

        auto chat_session = std::make_unique<llama_rs_chat_session>();

        try {
            chat_session->session = common_chat_session(templates->templates.get(), vocab, inputs);
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::exception & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        }

        chat_session->generation_prompt = common_chat_input_tokenize(vocab, chat_session->session.generation_prompt());
        for (const llama_token token : chat_session->generation_prompt.tokens) {
            if (token != LLAMA_TOKEN_NULL) {
                chat_session->generation_prompt_tokens.push_back(token);
            }
        }
        chat_session->session.apply_sampling(chat_session->sampling);
        chat_session->preserved_tokens.assign(
            chat_session->sampling.preserved_tokens.begin(), chat_session->sampling.preserved_tokens.end());

        *out_session = chat_session.release();

        return LLAMA_RS_CHAT_SESSION_CREATE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_SESSION_CREATE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_session_free(llama_rs_chat_session_handle session) -> void {
    const std::unique_ptr<llama_rs_chat_session> reclaimed(session);
}

extern "C" auto llama_rs_chat_session_prompt_part_count(llama_rs_chat_session_handle session) -> size_t {
    return session->session.prompt().parts.size();
}

extern "C" auto llama_rs_chat_session_prompt_part_text(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(session->session.prompt().parts.at(index).text, out_length);
}

extern "C" auto llama_rs_chat_session_prompt_part_is_input(llama_rs_chat_session_handle session, size_t index)
    -> bool {
    return session->session.prompt().parts.at(index).is_input;
}

extern "C" auto llama_rs_chat_session_grammar(llama_rs_chat_session_handle session, size_t * out_length)
    -> const char * {
    return llama_rs_string_view(session->session.grammar(), out_length);
}

extern "C" auto llama_rs_chat_session_grammar_lazy(llama_rs_chat_session_handle session) -> bool {
    return session->sampling.grammar_lazy;
}

extern "C" auto llama_rs_chat_session_grammar_trigger_count(llama_rs_chat_session_handle session) -> size_t {
    return session->sampling.grammar_triggers.size();
}

extern "C" auto llama_rs_chat_session_grammar_trigger_type(llama_rs_chat_session_handle session, size_t index)
    -> llama_rs_chat_grammar_trigger_type {
    switch (session->sampling.grammar_triggers.at(index).type) {
        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
            return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_TOKEN;
        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
            return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_WORD;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
            return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN;
        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
            return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
    }

    return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
}

extern "C" auto llama_rs_chat_session_grammar_trigger_value(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(session->sampling.grammar_triggers.at(index).value, out_length);
}

extern "C" auto llama_rs_chat_session_grammar_trigger_token(llama_rs_chat_session_handle session, size_t index)
    -> llama_token {
    return session->sampling.grammar_triggers.at(index).token;
}

extern "C" auto llama_rs_chat_session_preserved_token_count(llama_rs_chat_session_handle session) -> size_t {
    return session->preserved_tokens.size();
}

extern "C" auto llama_rs_chat_session_preserved_token(llama_rs_chat_session_handle session, size_t index)
    -> llama_token {
    return session->preserved_tokens.at(index);
}

extern "C" auto llama_rs_chat_session_generation_prompt(llama_rs_chat_session_handle session, size_t * out_length)
    -> const char * {
    return llama_rs_string_view(session->session.generation_prompt(), out_length);
}

extern "C" auto llama_rs_chat_session_generation_prompt_token_count(llama_rs_chat_session_handle session) -> size_t {
    return session->generation_prompt_tokens.size();
}

extern "C" auto llama_rs_chat_session_generation_prompt_token(llama_rs_chat_session_handle session, size_t index)
    -> llama_token {
    return session->generation_prompt_tokens.at(index);
}

extern "C" auto llama_rs_chat_session_thinking_start_tag(llama_rs_chat_session_handle session, size_t * out_length)
    -> const char * {
    return llama_rs_string_view(session->session.thinking_start_tag(), out_length);
}

extern "C" auto llama_rs_chat_session_thinking_end_tag_count(llama_rs_chat_session_handle session) -> size_t {
    return session->session.thinking_end_tags().size();
}

extern "C" auto llama_rs_chat_session_thinking_end_tag(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(session->session.thinking_end_tags().at(index), out_length);
}

extern "C" auto llama_rs_chat_session_parse(
    llama_rs_chat_session_handle session,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens,
    bool is_partial,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error) -> llama_rs_chat_session_parse_status {
    if (out_handle != nullptr) {
        *out_handle = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (session == nullptr) {
        return LLAMA_RS_CHAT_SESSION_PARSE_NULL_SESSION_ARG;
    }
    if (text == nullptr) {
        return LLAMA_RS_CHAT_SESSION_PARSE_NULL_TEXT_ARG;
    }
    if (byte_tokens == nullptr) {
        return LLAMA_RS_CHAT_SESSION_PARSE_NULL_BYTE_TOKENS_ARG;
    }
    if (out_handle == nullptr) {
        return LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_HANDLE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_SESSION_PARSE_NULL_OUT_ERROR_ARG;
    }

    try {
        common_chat_parser_params parser_params;
        parser_params.format = session->session.format();
        parser_params.generation_prompt = session->generation_prompt;

        auto parsed = std::make_unique<llama_rs_parsed_chat>();

        try {
            parsed->message = common_chat_peg_parse(
                session->session.parser(), chat_input_of(text, text_length, byte_tokens), is_partial, parser_params);
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::runtime_error & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_PARSE_OUTPUT_DOES_NOT_MATCH_FORMAT,
                LLAMA_RS_CHAT_SESSION_PARSE_ERROR_STRING_ALLOCATION_FAILED);
        }

        *out_handle = parsed.release();

        return LLAMA_RS_CHAT_SESSION_PARSE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_SESSION_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_SESSION_PARSE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_session_classify(
    llama_rs_chat_session_handle session,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens,
    bool is_partial,
    llama_rs_chat_spans_handle * out_spans,
    char ** out_error) -> llama_rs_chat_session_classify_status {
    if (out_spans != nullptr) {
        *out_spans = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (session == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_SESSION_ARG;
    }
    if (text == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_TEXT_ARG;
    }
    if (byte_tokens == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_BYTE_TOKENS_ARG;
    }
    if (out_spans == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_SPANS_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_SESSION_CLASSIFY_NULL_OUT_ERROR_ARG;
    }

    try {
        common_chat_input parse_input = chat_input_of(text, text_length, byte_tokens);
        parse_input.prepend(session->generation_prompt);

        const size_t generation_prompt_length = session->generation_prompt.size();

        common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_LENIENT;
        if (!is_partial) {
            flags |= COMMON_PEG_PARSE_FLAG_COMPLETE;
        }

        common_peg_parse_context context(std::move(parse_input.text), std::move(parse_input.tokens), flags);
        const common_peg_arena & parser = session_parser(session->session);
        const common_peg_parse_result result = parser.parse(context);

        if (result.fail() && (!is_partial || result.end == 0)) {
            return LLAMA_RS_CHAT_SESSION_CLASSIFY_OUTPUT_DOES_NOT_MATCH_FORMAT;
        }

        auto spans = std::make_unique<llama_rs_chat_spans>();
        spans->parsed_length = result.end > generation_prompt_length ? result.end - generation_prompt_length : 0;

        context.ast.visit(result, [&](const common_peg_ast_node & node) -> void {
            const auto kind = span_kind_of_node(parser, node);
            if (!kind || node.end <= generation_prompt_length) {
                return;
            }
            spans->spans.push_back({
                *kind,
                node.start > generation_prompt_length ? node.start - generation_prompt_length : 0,
                node.end - generation_prompt_length,
            });
        });

        *out_spans = spans.release();

        return LLAMA_RS_CHAT_SESSION_CLASSIFY_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_SESSION_CLASSIFY_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_SESSION_CLASSIFY_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_spans_parsed_length(llama_rs_chat_spans_handle spans) -> size_t {
    return spans->parsed_length;
}

extern "C" auto llama_rs_chat_spans_count(llama_rs_chat_spans_handle spans) -> size_t {
    return spans->spans.size();
}

extern "C" auto llama_rs_chat_spans_kind(llama_rs_chat_spans_handle spans, size_t index) -> llama_rs_chat_span_kind {
    return spans->spans.at(index).kind;
}

extern "C" auto llama_rs_chat_spans_start(llama_rs_chat_spans_handle spans, size_t index) -> size_t {
    return spans->spans.at(index).start;
}

extern "C" auto llama_rs_chat_spans_end(llama_rs_chat_spans_handle spans, size_t index) -> size_t {
    return spans->spans.at(index).end;
}

extern "C" auto llama_rs_chat_spans_free(llama_rs_chat_spans_handle spans) -> void {
    const std::unique_ptr<llama_rs_chat_spans> reclaimed(spans);
}

extern "C" auto llama_rs_regex_escape(
    const char * text,
    size_t text_length,
    char ** out_string,
    char ** out_error) -> llama_rs_regex_escape_status {
    if (out_string != nullptr) {
        *out_string = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (text == nullptr) {
        return LLAMA_RS_REGEX_ESCAPE_NULL_TEXT_ARG;
    }
    if (out_string == nullptr) {
        return LLAMA_RS_REGEX_ESCAPE_NULL_OUT_STRING_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_REGEX_ESCAPE_NULL_OUT_ERROR_ARG;
    }

    try {
        *out_string = llama_rs_dup_string(regex_escape(std::string(text, text_length)));
        if (*out_string == nullptr) {
            return LLAMA_RS_REGEX_ESCAPE_ERROR_STRING_ALLOCATION_FAILED;
        }

        return LLAMA_RS_REGEX_ESCAPE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_REGEX_ESCAPE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_REGEX_ESCAPE_ERROR_STRING_ALLOCATION_FAILED);
    }
}
