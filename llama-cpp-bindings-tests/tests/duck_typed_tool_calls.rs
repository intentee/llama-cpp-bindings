use anyhow::Result;
use llama_cpp_bindings::ChatOutputCompleteness;
use llama_cpp_bindings_tests::get_weather_tools_json::GET_WEATHER_TOOLS_JSON;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

fn assert_duck_types_paris_weather_call(fixture: &LlamaFixture<'_>, answer: &str) -> Result<()> {
    let session = user_turn_chat_session(
        fixture.model,
        "What is the weather in Paris?",
        GET_WEATHER_TOOLS_JSON,
        false,
    )?;

    assert!(!session.supports_tool_calls());
    assert_eq!(
        recognized_answer(&session, answer, ChatOutputCompleteness::Finished)?.tool_calls,
        [paris_weather_tool_call()]
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
fn deepseek_r1_8b_duck_types_gemma_paired_quote(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_duck_types_paris_weather_call(
        fixture,
        "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}",
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_glm_key_value_tags(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_duck_types_paris_weather_call(
        fixture,
        "<tool_call>get_weather<arg_key>location</arg_key><arg_value>Paris</arg_value></tool_call>",
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_mistral_bracketed_json(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_duck_types_paris_weather_call(
        fixture,
        r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#,
    )
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn deepseek_r1_8b_duck_types_qwen_xml(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_duck_types_paris_weather_call(
        fixture,
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>",
    )
}
