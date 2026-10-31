use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::chat_message_parse_outcome::ChatMessageParseOutcome;
use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::parse_chat_message_error::ParseChatMessageError;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_bindings_types::ParsedToolCall;
use llama_cpp_fault_injection::ArmedFault;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_pure_content_response(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = closed_reasoning_chat_session(fixture.model, "hello", "[]").parse(
        &ChatOutput::from("hello world"),
        ChatOutputCompleteness::Finished,
    )?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for plain content; got Unrecognized");
    };
    assert_eq!(parsed.tool_calls, Vec::<ParsedToolCall>::new());
    assert!(!parsed.is_empty());
    assert!(parsed.content.contains("hello world"));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_reasoning_section_into_reasoning_content(fixture: &LlamaFixture<'_>) -> Result<()> {
    let input = "<think>step one, step two</think>\n\nactual response";
    let outcome = user_turn_chat_session(fixture.model, "hello", "[]", false)?
        .parse(&ChatOutput::from(input), ChatOutputCompleteness::Finished)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for reasoning section; got Unrecognized");
    };
    assert!(
        parsed.reasoning_content.contains("step") || parsed.content.contains("step"),
        "neither content nor reasoning contains 'step'; content={:?} reasoning={:?}",
        parsed.content,
        parsed.reasoning_content
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_empty_input_yields_empty_message(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = user_turn_chat_session(fixture.model, "hello", "[]", false)?
        .parse(&ChatOutput::from(""), ChatOutputCompleteness::Finished)?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("expected Recognized for empty input; got Unrecognized");
    };
    assert_eq!(parsed.tool_calls, Vec::<ParsedToolCall>::new());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_an_answer_with_a_nul_byte_as_content(fixture: &LlamaFixture<'_>) -> Result<()> {
    let outcome = user_turn_chat_session(fixture.model, "hello", "[]", false)?.parse(
        &ChatOutput::from("hello\0world"),
        ChatOutputCompleteness::Finished,
    )?;

    let ChatMessageParseOutcome::Recognized(parsed) = outcome else {
        bail!("an answer with a NUL byte must be recognized as content");
    };

    assert_eq!(parsed.content, "hello\0world");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_chat_session_parses_every_answer_it_is_given(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", "[]", false)?;

    for answer in ["the first answer", "the second answer"] {
        let ChatMessageParseOutcome::Recognized(parsed) =
            session.parse(&ChatOutput::from(answer), ChatOutputCompleteness::Finished)?
        else {
            bail!("expected {answer:?} to be recognized as content");
        };

        assert_eq!(parsed.content, answer);
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn every_parsed_field_refuses_text_that_is_not_utf8(fixture: &LlamaFixture<'_>) {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let answer = ChatOutput::from(
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>",
    );
    let not_utf8 = String::from_utf8(vec![0xff])
        .expect_err("a lone 0xff byte is not UTF-8")
        .utf8_error();
    let fault_points = [
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_PARSED_CHAT_CONTENT_NOT_UTF8,
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_PARSED_CHAT_REASONING_CONTENT_NOT_UTF8,
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_ID_NOT_UTF8,
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_NAME_NOT_UTF8,
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_PARSED_CHAT_TOOL_CALL_ARGUMENTS_NOT_UTF8,
    ];

    assert_eq!(
        fault_points.map(|fault_point| {
            let _fault = ArmedFault::at(fault_point);

            session
                .parse(&answer, ChatOutputCompleteness::Finished)
                .err()
        }),
        fault_points.map(|_fault_point| Some(ParseChatMessageError::ParsedTextNotUtf8(not_utf8)))
    );
}
