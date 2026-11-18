use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::not_utf8::not_utf8;
use llama_cpp_bindings_tests::weather_chat_session::weather_chat_session;
use llama_cpp_chat::chat_error::ChatError;
use llama_cpp_chat::chat_tool_choice::ChatToolChoice;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_fault_injection::injected_fault_message::injected_fault_message;
use llama_cpp_sampler::sampler_error::SamplerError;
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
fn the_grammar_sampler_refuses_a_grammar_that_is_not_utf8(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let session = weather_chat_session(model, ChatToolChoice::Required)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_CHAT_SESSION_GRAMMAR_NOT_UTF8);

    ensure!(session.grammar_sampler(model).err() == not_utf8().map(ChatError::SessionTextNotUtf8));

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
