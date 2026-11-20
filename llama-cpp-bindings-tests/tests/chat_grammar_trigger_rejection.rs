use anyhow::Context as _;
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
fn grammar_triggers_refuse_a_trigger_value_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let expected_utf8_error = not_utf8().context("a lone 0xff byte is not UTF-8")?;
    let session = weather_chat_session(model, ChatToolChoice::Auto)?;
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_VALUE_NOT_UTF8,
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
        llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_TRIGGER_TYPE_UNRECOGNIZED,
    );

    ensure!(session.grammar_triggers().err() == Some(ChatError::GrammarTriggerTypeUnrecognized));

    Ok(())
}
