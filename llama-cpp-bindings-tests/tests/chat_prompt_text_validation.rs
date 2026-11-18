use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::not_utf8::not_utf8;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

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
    let expected_utf8_error = not_utf8();
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_PROMPT_PART_TEXT_NOT_UTF8,
    );

    ensure!(
        [
            session.prompt_parts().err(),
            session.prompt().err(),
            session.tokenize_prompt(model).err(),
        ] == [(); 3].map(|()| expected_utf8_error.map(ChatError::SessionTextNotUtf8))
    );

    Ok(())
}
