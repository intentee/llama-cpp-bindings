#include "wrapper_chat_session.h"
#include "wrapper_fault_injection.h"
#include "wrapper_chat_templates.h"
#include "wrapper_chat_templates_handle.h"
#include "wrapper_parsed_chat.h"
#include "wrapper_parsed_chat_handle.h"
#include "wrapper_report_current_exception.h"
#include "wrapper_utils.h"

#include <common/chat-peg-parser.h>
#include <common/chat.h>
#include <common/common.h>
#include <common/json.h>
#include <common/peg-parser.h>
#include <llama.h>

#include <cstddef>
#include <exception>
#include <gsl/span>
#include <memory>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

struct llama_rs_chat_session {
    std::shared_ptr<const common_chat_session> session;
    common_chat_input generation_prompt;
    std::vector<llama_token> generation_prompt_tokens;
    common_params_sampling sampling;
    std::vector<llama_token> preserved_tokens;
    std::unordered_set<std::string> trigger_rules;
};

struct llama_rs_chat_reading {
    std::shared_ptr<const common_chat_session> session;
    bool is_partial = false;
    size_t generation_prompt_length = 0;
    common_peg_parse_context context;
    common_peg_parse_result result;
    bool matches_format = false;
    size_t parsed_length = 0;
    std::vector<llama_rs_chat_span> spans;
    std::string format_name;
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

    throw std::invalid_argument("unrecognized llama_rs_chat_tool_choice value");
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

    throw std::invalid_argument("unrecognized llama_rs_chat_assistant_turn value");
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

auto trigger_rules_of(const common_peg_arena & parser) -> std::unordered_set<std::string> {
    std::unordered_set<std::string> trigger_rules;

    for (common_peg_parser_id parser_id = 0; parser_id < parser.size(); ++parser_id) {
        const auto * rule_parser = std::get_if<common_peg_rule_parser>(&parser.get(parser_id));
        if (rule_parser != nullptr && rule_parser->trigger && !rule_parser->name.empty() &&
            parser.get_rule(rule_parser->name) == parser_id) {
            trigger_rules.insert(rule_parser->name);
        }
    }

    return trigger_rules;
}

auto span_kind_of_node(const std::unordered_set<std::string> & trigger_rules, const common_peg_ast_node & node)
    -> std::optional<llama_rs_chat_span_kind> {
    if (const auto kind = span_kind_of_tag(node.tag)) {
        return kind;
    }
    if (trigger_rules.find(node.rule) != trigger_rules.end()) {
        return LLAMA_RS_CHAT_SPAN_KIND_TOOL_CALL_MARKER;
    }

    return std::nullopt;
}

auto chat_input_of(
    const common_chat_input & prefix,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens) -> common_chat_input {
    const gsl::span<const llama_token> tokens(byte_tokens, text_length);
    common_chat_input input;
    input.text.reserve(prefix.text.size() + text_length);
    input.text.append(prefix.text).append(text, text_length);
    input.tokens.reserve(prefix.tokens.size() + text_length);
    input.tokens.insert(input.tokens.end(), prefix.tokens.begin(), prefix.tokens.end());
    input.tokens.insert(input.tokens.end(), tokens.begin(), tokens.end());

    return input;
}

auto finished_output_of(const llama_rs_chat_reading & reading) -> common_chat_input {
    common_chat_input output;
    output.text = reading.context.input.substr(reading.generation_prompt_length);
    output.tokens.assign(
        reading.context.tokens.begin() + static_cast<std::ptrdiff_t>(reading.generation_prompt_length),
        reading.context.tokens.end());

    return output;
}

auto finished_output_parser_params_of(const llama_rs_chat_reading & reading) -> common_chat_parser_params {
    common_chat_parser_params parser_params;
    parser_params.format = reading.session->format();
    parser_params.generation_prompt.text = reading.context.input.substr(0, reading.generation_prompt_length);
    parser_params.generation_prompt.tokens.assign(
        reading.context.tokens.begin(),
        reading.context.tokens.begin() + static_cast<std::ptrdiff_t>(reading.generation_prompt_length));

    return parser_params;
}

auto mapper_of(common_chat_format format, common_chat_msg & message) -> std::unique_ptr<common_chat_peg_mapper> {
    if (format == COMMON_CHAT_FORMAT_PEG_GEMMA4) {
        return std::make_unique<common_chat_peg_gemma4_mapper>(message);
    }
    if (format == COMMON_CHAT_FORMAT_PEG_MINIMAX_M3) {
        return std::make_unique<common_chat_peg_minimax_m3_mapper>(message);
    }

    return std::make_unique<common_chat_peg_mapper>(message);
}
}  // namespace

extern "C" auto llama_rs_chat_session_create(
    llama_rs_chat_templates_handle templates,
    const struct llama_vocab * vocab,
    const char * messages_json,
    size_t messages_json_length,
    const char * tools_json,
    size_t tools_json_length,
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

        inputs.messages = common_chat_msgs_parse_oaicompat(
            common_json::parse(std::string(messages_json, messages_json_length)));

        inputs.tools = common_chat_tools_parse_oaicompat(
            common_json::parse(std::string(tools_json, tools_json_length)));

        inputs.add_generation_prompt = options.assistant_turn == LLAMA_RS_CHAT_ASSISTANT_TURN_NEW;
        inputs.continue_final_message = continuation_of(options.assistant_turn);
        inputs.enable_thinking = options.enable_thinking;
        inputs.parallel_tool_calls = options.parallel_tool_calls;
        inputs.tool_choice = tool_choice_of(options.tool_choice);
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;

        auto chat_session = std::make_unique<llama_rs_chat_session>();

        try {
            chat_session->session = std::make_shared<const common_chat_session>(templates->templates.get(), vocab, inputs);
        } catch (const std::bad_alloc &) {
            throw;
        } catch (const std::exception & err) {
            return llama_rs_report_exception_message(
                out_error, err.what(), LLAMA_RS_CHAT_SESSION_CREATE_TEMPLATE_CANNOT_SERVE_REQUEST,
                LLAMA_RS_CHAT_SESSION_CREATE_ERROR_STRING_ALLOCATION_FAILED);
        }

        chat_session->generation_prompt = common_chat_input_tokenize(vocab, chat_session->session->generation_prompt());
        for (const llama_token token : chat_session->generation_prompt.tokens) {
            if (token != LLAMA_TOKEN_NULL) {
                chat_session->generation_prompt_tokens.push_back(token);
            }
        }
        chat_session->session->apply_sampling(chat_session->sampling);
        chat_session->preserved_tokens.assign(
            chat_session->sampling.preserved_tokens.begin(), chat_session->sampling.preserved_tokens.end());
        chat_session->trigger_rules = trigger_rules_of(chat_session->session->parser());

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
    std::unique_ptr<llama_rs_chat_session>(session).reset();
}

extern "C" auto llama_rs_chat_session_prompt_part_count(llama_rs_chat_session_handle session) -> size_t {
    return session->session->prompt().parts.size();
}

extern "C" auto llama_rs_chat_session_prompt_part_text(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_CHAT_SESSION_PROMPT_PART_TEXT_NOT_UTF8,
        session->session->prompt().parts.at(index).text,
        out_length);
}

extern "C" auto llama_rs_chat_session_prompt_part_is_input(llama_rs_chat_session_handle session, size_t index)
    -> bool {
    return session->session->prompt().parts.at(index).is_input;
}

extern "C" auto llama_rs_chat_session_grammar(llama_rs_chat_session_handle session, size_t * out_length)
    -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_NOT_UTF8, session->session->grammar(), out_length);
}

extern "C" auto llama_rs_chat_session_grammar_lazy(llama_rs_chat_session_handle session) -> bool {
    return session->sampling.grammar_lazy;
}

extern "C" auto llama_rs_chat_session_grammar_trigger_count(llama_rs_chat_session_handle session) -> size_t {
    return session->sampling.grammar_triggers.size();
}

extern "C" auto llama_rs_chat_session_grammar_trigger_type(llama_rs_chat_session_handle session, size_t index)
    -> llama_rs_chat_grammar_trigger_type {
    if (llama_rs_fault_injected(LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED)) {
        return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED;
    }

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

    return LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED;
}

extern "C" auto llama_rs_chat_session_grammar_trigger_value(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_VALUE_NOT_UTF8,
        session->sampling.grammar_triggers.at(index).value,
        out_length);
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
    return llama_rs_string_view(session->session->generation_prompt(), out_length);
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
    return llama_rs_string_view(session->session->thinking_start_tag(), out_length);
}

extern "C" auto llama_rs_chat_session_thinking_end_tag_count(llama_rs_chat_session_handle session) -> size_t {
    return session->session->thinking_end_tags().size();
}

extern "C" auto llama_rs_chat_session_thinking_end_tag(
    llama_rs_chat_session_handle session,
    size_t index,
    size_t * out_length) -> const char * {
    return llama_rs_string_view(session->session->thinking_end_tags().at(index), out_length);
}

extern "C" auto llama_rs_chat_session_read(
    llama_rs_chat_session_handle session,
    const char * text,
    size_t text_length,
    const llama_token * byte_tokens,
    bool is_partial,
    llama_rs_chat_reading_handle * out_reading,
    char ** out_error) -> llama_rs_chat_session_read_status {
    if (out_reading != nullptr) {
        *out_reading = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (session == nullptr) {
        return LLAMA_RS_CHAT_SESSION_READ_NULL_SESSION_ARG;
    }
    if (text == nullptr) {
        return LLAMA_RS_CHAT_SESSION_READ_NULL_TEXT_ARG;
    }
    if (byte_tokens == nullptr) {
        return LLAMA_RS_CHAT_SESSION_READ_NULL_BYTE_TOKENS_ARG;
    }
    if (out_reading == nullptr) {
        return LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_READING_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_SESSION_READ_NULL_OUT_ERROR_ARG;
    }

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

        common_chat_input parse_input = chat_input_of(session->generation_prompt, text, text_length, byte_tokens);
        const size_t generation_prompt_length = session->generation_prompt.size();
        auto reading = std::make_unique<llama_rs_chat_reading>();

        reading->session = session->session;
        reading->is_partial = is_partial;
        reading->generation_prompt_length = generation_prompt_length;
        reading->context.input = std::move(parse_input.text);
        reading->context.tokens = std::move(parse_input.tokens);
        reading->context.flags = is_partial ? COMMON_PEG_PARSE_FLAG_LENIENT : COMMON_PEG_PARSE_FLAG_COMPLETE;

        reading->result = session->session->parser().parse(reading->context);
        reading->matches_format = is_partial
            ? !reading->result.fail() || reading->result.end > 0
            : !reading->result.fail() && reading->result.end == reading->context.input.size();

        if (!reading->matches_format) {
            reading->format_name = common_chat_format_name(session->session->format());
            *out_reading = reading.release();

            return LLAMA_RS_CHAT_SESSION_READ_OK;
        }

        reading->parsed_length =
            reading->result.end > generation_prompt_length ? reading->result.end - generation_prompt_length : 0;

        reading->context.ast.visit(reading->result, [&](const common_peg_ast_node & node) -> void {
            const auto kind = span_kind_of_node(session->trigger_rules, node);
            if (!kind || node.end <= generation_prompt_length) {
                return;
            }
            reading->spans.push_back({
                *kind,
                node.start > generation_prompt_length ? node.start - generation_prompt_length : 0,
                node.end - generation_prompt_length,
            });
        });

        *out_reading = reading.release();

        return LLAMA_RS_CHAT_SESSION_READ_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_SESSION_READ_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_SESSION_READ_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_reading_matches_format(llama_rs_chat_reading_handle reading) -> bool {
    return reading->matches_format;
}

extern "C" auto llama_rs_chat_reading_spans_view(llama_rs_chat_reading_handle reading) -> llama_rs_chat_spans_view {
    return {reading->spans.data(), reading->spans.size(), reading->parsed_length};
}

extern "C" auto llama_rs_chat_reading_format_name(llama_rs_chat_reading_handle reading, size_t * out_length)
    -> const char * {
    return llama_rs_text_view_unless_fault_injected(
        LLAMA_RS_FAULT_POINT_CHAT_READING_FORMAT_NAME_NOT_UTF8, reading->format_name, out_length);
}

extern "C" auto llama_rs_chat_reading_parse(
    llama_rs_chat_reading_handle reading,
    llama_rs_parsed_chat_handle * out_handle,
    char ** out_error) -> llama_rs_chat_reading_parse_status {
    if (out_handle != nullptr) {
        *out_handle = nullptr;
    }
    if (out_error != nullptr) {
        *out_error = nullptr;
    }
    if (reading == nullptr) {
        return LLAMA_RS_CHAT_READING_PARSE_NULL_READING_ARG;
    }
    if (out_handle == nullptr) {
        return LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_HANDLE_ARG;
    }
    if (out_error == nullptr) {
        return LLAMA_RS_CHAT_READING_PARSE_NULL_OUT_ERROR_ARG;
    }

    try {
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_CHAT_READING_PARSE_THROWS);

        auto parsed = std::make_unique<llama_rs_parsed_chat>();
        parsed->message.role = "assistant";

        if (reading->is_partial) {
            mapper_of(reading->session->format(), parsed->message)->from_ast(reading->context.ast, reading->result);
        } else {
            parsed->message = common_chat_peg_parse(
                reading->session->parser(), finished_output_of(*reading), false, finished_output_parser_params_of(*reading));
        }

        *out_handle = parsed.release();

        return LLAMA_RS_CHAT_READING_PARSE_OK;
    } catch (...) {
        return llama_rs_report_current_exception(
            out_error,
            LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_OUT_OF_MEMORY,
            LLAMA_RS_CHAT_READING_PARSE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            LLAMA_RS_CHAT_READING_PARSE_ERROR_STRING_ALLOCATION_FAILED);
    }
}

extern "C" auto llama_rs_chat_reading_free(llama_rs_chat_reading_handle reading) -> void {
    std::unique_ptr<llama_rs_chat_reading>(reading).reset();
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
        llama_rs_throw_if_fault_injected(LLAMA_RS_FAULT_POINT_REGEX_ESCAPE_THROWS);

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
