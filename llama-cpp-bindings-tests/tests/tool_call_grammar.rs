use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::chat_output_completeness::ChatOutputCompleteness;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::chat_answer_generation::ChatAnswerGeneration;
use llama_cpp_bindings_tests::paris_weather_tool_call::paris_weather_tool_call;
use llama_cpp_bindings_tests::recognized_answer::recognized_answer;
use llama_cpp_bindings_tests::user_turn_chat_session::user_turn_chat_session;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

const MAX_TOOL_CALL_TOKENS: i32 = 128;

fn generated_tool_call_answer(
    fixture: &LlamaFixture<'_>,
    session: &ChatSession<'_>,
) -> Result<String> {
    let grammar_sampler = session
        .grammar_sampler()?
        .context("a tool request must constrain generation with a grammar")?;
    let mut sampler = LlamaSampler::chain([grammar_sampler, LlamaSampler::greedy()?])?;
    let answer = ChatAnswerGeneration {
        fixture,
        session,
        sampler: &mut sampler,
        max_generated_tokens: MAX_TOOL_CALL_TOKENS,
    }
    .run();

    Ok(answer.outcome.generated.text().to_owned())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_chat_session_without_tools_needs_no_grammar_sampler(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = user_turn_chat_session(fixture.model, "Hello!", "[]", false)?;

    assert!(session.grammar_sampler()?.is_none());

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
fn a_tool_choice_of_none_needs_no_grammar_sampler(fixture: &LlamaFixture<'_>) -> Result<()> {
    let session = weather_chat_session(fixture.model, ChatToolChoice::None);

    assert!(session.grammar_sampler()?.is_none());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen36_required_tool_call_grammar_constrains_the_answer_to_a_tool_call(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Required);
    let answer = generated_tool_call_answer(fixture, &session)?;

    assert!(!session.grammar_is_lazy());
    assert_eq!(
        recognized_answer(&session, &answer, ChatOutputCompleteness::Finished).tool_calls,
        [paris_weather_tool_call()]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen36_automatic_tool_call_grammar_constrains_the_tool_call_after_its_trigger(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(fixture.model, ChatToolChoice::Auto);
    let answer = generated_tool_call_answer(fixture, &session)?;

    assert!(session.grammar_is_lazy());
    assert_eq!(
        recognized_answer(&session, &answer, ChatOutputCompleteness::Finished).tool_calls,
        [paris_weather_tool_call()]
    );

    Ok(())
}
