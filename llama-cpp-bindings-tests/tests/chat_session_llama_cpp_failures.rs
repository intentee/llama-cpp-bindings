use std::str::Utf8Error;

use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::chat_session_error::ChatSessionError;
use llama_cpp_bindings::error::chat_templates_error::ChatTemplatesError;
use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::error::parse_chat_message_error::ParseChatMessageError;
use llama_cpp_bindings::error::sampler_accept_error::SamplerAcceptError;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn not_utf8() -> Utf8Error {
    String::from_utf8(vec![0xff])
        .expect_err("a lone 0xff byte is not UTF-8")
        .utf8_error()
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_chat_session_needs_a_model_with_a_chat_template(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        user_turn_chat_session(fixture.model, "hello", "[]", false)
            .err()
            .map(|failure| failure.downcast::<ChatSessionError>().ok()),
        Some(Some(ChatSessionError::Templates(
            ChatTemplatesError::ModelHasNoChatTemplate
        )))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn every_prompt_reader_refuses_prompt_text_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_PROMPT_PART_TEXT_NOT_UTF8,
    );

    assert_eq!(
        [
            session.prompt_parts().err(),
            session.prompt().err(),
            session.tokenize_prompt().err(),
        ],
        [(); 3].map(|()| Some(ChatSessionError::SessionTextNotUtf8(not_utf8())))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn grammar_triggers_refuse_a_trigger_value_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_VALUE_NOT_UTF8,
    );

    assert_eq!(
        [
            session.grammar_triggers().err(),
            session.grammar_sampler().err(),
        ],
        [(); 2].map(|()| Some(ChatSessionError::SessionTextNotUtf8(not_utf8())))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn grammar_triggers_refuse_a_trigger_type_the_bindings_do_not_recognize(
    fixture: &LlamaFixture<'_>,
) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED,
    );

    assert_eq!(
        session.grammar_triggers().err(),
        Some(ChatSessionError::FfiStatus(FfiStatusError {
            operation: "llama_rs_chat_session_grammar_trigger_type",
            code: i64::from(
                llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED
            ),
        }))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_refuses_a_grammar_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Required);
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_NOT_UTF8);

    assert_eq!(
        session.grammar_sampler().err(),
        Some(ChatSessionError::SessionTextNotUtf8(not_utf8()))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_grammar_llama_cpp_cannot_load(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Required);
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_THROWS);

    assert!(!session.grammar_is_lazy());
    assert_eq!(
        session.grammar_sampler().err(),
        Some(ChatSessionError::Grammar(GrammarError::Reported {
            message: injected_fault_message(),
        }))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_generation_prompt_token_the_grammar_cannot_accept(
    fixture: &LlamaFixture<'_>,
) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Required);
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_ACCEPT_THROWS);

    assert_ne!(session.generation_prompt_tokens(), []);
    assert_eq!(
        session.grammar_sampler().err(),
        Some(ChatSessionError::GrammarPrefill(
            SamplerAcceptError::GrammarStateCorrupted {
                message: injected_fault_message(),
            }
        ))
    );
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q2_k.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn the_grammar_sampler_reports_a_trigger_word_llama_cpp_cannot_escape(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_REGEX_ESCAPE_THROWS);

    assert!(session.grammar_is_lazy());
    assert_eq!(
        session.grammar_sampler().err(),
        Some(ChatSessionError::Grammar(GrammarError::Reported {
            message: injected_fault_message(),
        }))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_lazy_grammar_llama_cpp_cannot_load(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THROWS,
    );

    assert_eq!(
        session.grammar_sampler().err(),
        Some(ChatSessionError::Grammar(GrammarError::Reported {
            message: injected_fault_message(),
        }))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parsing_a_duck_typed_tool_call_reports_a_parser_llama_cpp_fails(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_PARSE_THROWS);

    assert!(!session.supports_tool_calls());
    assert_eq!(
        session
            .parse(
                &ChatOutput::from(
                    "Checking.<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>"
                ),
                ChatOutputCompleteness::Finished,
            )
            .err(),
        Some(ParseChatMessageError::Reported {
            message: injected_fault_message(),
        })
    );
}
