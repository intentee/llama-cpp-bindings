use anyhow::Result;

use llama_cpp_bindings::chat_output::ChatOutput;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::error::parse_chat_message_error::ParseChatMessageError;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_bindings_types::parsed_chat_message::ParsedChatMessage;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const GENERATED_REASONING: &str = "step one, step two";
const GENERATED_CONTENT: &str = "actual response";

fn parsed_content_after_closed_reasoning(fixture: &LlamaFixture<'_>) -> ParsedChatMessage {
    recognized_answer(
        &closed_reasoning_chat_session(fixture.model, "hello", &[]),
        "hello world",
        ChatOutputCompleteness::Finished,
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn templates_that_keep_prefill_whitespace_parse_it_into_reasoning_and_content(
    fixture: &LlamaFixture<'_>,
) {
    assert_eq!(
        parsed_content_after_closed_reasoning(fixture),
        ParsedChatMessage::new(
            "\n\nhello world".to_owned(),
            "\nThe question is simple, so I answer it directly.\n".to_owned(),
            Vec::new()
        )
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen_templates_parse_trimmed_prefill_into_reasoning_and_content(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        parsed_content_after_closed_reasoning(fixture),
        ParsedChatMessage::new(
            "hello world".to_owned(),
            "The question is simple, so I answer it directly.\n".to_owned(),
            Vec::new()
        )
    );
}

fn reasoning_parsed_before_the_end_tag(fixture: &LlamaFixture<'_>) -> Result<ParsedChatMessage> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], true)?;

    Ok(recognized_answer(
        &session,
        &format!("{GENERATED_REASONING}</think>{GENERATED_CONTENT}"),
        ChatOutputCompleteness::Finished,
    ))
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_parses_the_reasoning_after_the_newline_its_generation_prompt_opens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    assert_eq!(
        reasoning_parsed_before_the_end_tag(fixture)?,
        ParsedChatMessage::new(
            GENERATED_CONTENT.to_owned(),
            format!("\n{GENERATED_REASONING}"),
            Vec::new()
        )
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_the_reasoning_before_the_end_tag_apart_from_the_content(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    assert_eq!(
        reasoning_parsed_before_the_end_tag(fixture)?,
        ParsedChatMessage::new(
            GENERATED_CONTENT.to_owned(),
            GENERATED_REASONING.to_owned(),
            Vec::new()
        )
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
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_an_empty_answer_into_an_empty_message(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;

    assert_eq!(
        recognized_answer(&session, "", ChatOutputCompleteness::Finished),
        ParsedChatMessage::default()
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn parses_an_answer_with_a_nul_byte_as_content(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;

    assert_eq!(
        recognized_answer(&session, "hello\0world", ChatOutputCompleteness::Finished).content,
        "hello\0world"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_chat_session_parses_every_answer_it_is_given(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "hello", &[], false)?;

    for answer in ["the first answer", "the second answer"] {
        assert_eq!(
            recognized_answer(&session, answer, ChatOutputCompleteness::Finished).content,
            answer
        );
    }

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
            let reading = session
                .read(&answer, ChatOutputCompleteness::Finished)
                .expect("the weather answer must be read");
            let _fault = ArmedFault::at(fault_point);

            reading.parse().err().cloned()
        }),
        fault_points.map(|_fault_point| Some(ParseChatMessageError::ParsedTextNotUtf8(not_utf8)))
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_reading_maps_its_parse_once_and_keeps_it(fixture: &LlamaFixture<'_>) -> Result<()> {
    let reading = user_turn_chat_session(fixture.model, "hello", &[], false)?.read(
        &ChatOutput::from("Hello there."),
        ChatOutputCompleteness::Finished,
    )?;
    let first_parse = reading.parse().cloned();
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_READING_PARSE_THROWS);

    assert_eq!(reading.parse().cloned(), first_parse);

    Ok(())
}
