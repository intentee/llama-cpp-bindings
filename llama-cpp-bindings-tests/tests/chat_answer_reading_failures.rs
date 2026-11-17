use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::not_utf8::not_utf8;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_output::ChatOutput;
use llama_cpp_chat::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const MALFORMED_QWEN_XML_TOOL_CALL: &str = "<tool_call>garbage</tool_call>";

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
