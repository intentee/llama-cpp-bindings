use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::str::Utf8Error;

use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const MALFORMED_QWEN_XML_TOOL_CALL: &str = "<tool_call>garbage</tool_call>";

fn not_utf8() -> Result<Utf8Error> {
    Ok(String::from_utf8(vec![0xff])
        .err()
        .context("a lone 0xff byte is not UTF-8")?
        .utf8_error())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_model_without_a_chat_template_has_no_chat_templates(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        model.chat_templates().err(),
        Some(ModelError::ModelHasNoChatTemplate)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn every_prompt_reader_refuses_prompt_text_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let expected_utf8_error = not_utf8()?;
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_PROMPT_PART_TEXT_NOT_UTF8,
    );

    ensure!(
        [
            session.prompt_parts().err(),
            session.prompt().err(),
            session.tokenize_prompt(model).err(),
        ] == [(); 3].map(|()| Some(ChatError::SessionTextNotUtf8(expected_utf8_error)))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn grammar_triggers_refuse_a_trigger_value_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let expected_utf8_error = not_utf8()?;
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_VALUE_NOT_UTF8,
    );

    ensure!(
        [
            session.grammar_triggers().err(),
            session.grammar_sampler(model).err(),
        ] == [(); 2].map(|()| Some(ChatError::SessionTextNotUtf8(expected_utf8_error)))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn grammar_triggers_refuse_a_trigger_type_the_bindings_do_not_recognize(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED,
    );

    ensure!(
        session.grammar_triggers().err()
            == Some(ChatError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_chat_session_grammar_trigger_type",
                code: i64::from(
                    llama_cpp_bindings_sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED
                ),
            }))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_refuses_a_grammar_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Required)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_NOT_UTF8);

    ensure!(
        session.grammar_sampler(model).err() == Some(ChatError::SessionTextNotUtf8(not_utf8()?))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_grammar_llama_cpp_cannot_load(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Required)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_THROWS);

    ensure!(!session.grammar_is_lazy());
    ensure!(
        session.grammar_sampler(model).err()
            == Some(ChatError::Grammar(SamplerError::GrammarReported {
                message: injected_fault_message()?,
            }))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_generation_prompt_token_the_grammar_cannot_accept(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Required)?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_ACCEPT_THROWS);

    ensure!(session.generation_prompt_tokens() != []);
    ensure!(
        session.grammar_sampler(model).err()
            == Some(ChatError::GrammarPrefill(
                SamplerError::GrammarStateCorrupted {
                    message: injected_fault_message()?,
                }
            ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q2_k.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn the_grammar_sampler_reports_a_trigger_word_llama_cpp_cannot_escape(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_REGEX_ESCAPE_THROWS);

    ensure!(session.grammar_is_lazy());
    ensure!(
        session.grammar_sampler(model).err()
            == Some(ChatError::Grammar(SamplerError::GrammarReported {
                message: injected_fault_message()?,
            }))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_grammar_sampler_reports_a_lazy_grammar_llama_cpp_cannot_load(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_LAZY_PATTERNS_THROWS,
    );

    ensure!(
        session.grammar_sampler(model).err()
            == Some(ChatError::Grammar(SamplerError::GrammarReported {
                message: injected_fault_message()?,
            }))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn reading_an_answer_reports_a_parse_llama_cpp_fails(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_READ_THROWS);

    ensure!(
        session
            .read(
                &ChatOutput::from("Checking."),
                ChatOutputCompleteness::Finished,
            )
            .err()
            == Some(ChatError::ParseReported {
                message: injected_fault_message()?,
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parsing_a_reading_reports_a_mapping_llama_cpp_fails(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let reading = weather_chat_session(model, ChatToolChoice::Auto)?
        .read(
            &ChatOutput::from("Checking."),
            ChatOutputCompleteness::Finished,
        )
        .context("the answer must be read")?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_READING_PARSE_THROWS);

    ensure!(
        reading.parse().err()
            == Some(&ChatError::ParseReported {
                message: injected_fault_message()?,
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn reading_an_answer_that_does_not_match_the_format_refuses_a_format_name_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_READING_FORMAT_NAME_NOT_UTF8,
    );

    ensure!(
        session
            .read(
                &ChatOutput::from(MALFORMED_QWEN_XML_TOOL_CALL),
                ChatOutputCompleteness::Finished,
            )
            .err()
            == Some(ChatError::FormatNameNotUtf8(not_utf8()?))
    );

    Ok(())
}
