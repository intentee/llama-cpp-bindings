use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use std::ffi::CStr;

use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_bindings_tests::yes_or_no_regex::YES_OR_NO_REGEX;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn out_of_vocabulary_failure() -> String {
    SamplerError::GrammarInternalParserError {
        message: "token id 2147483646 out of range".to_owned(),
    }
    .to_string()
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn an_llguidance_sampler_names_itself_llguidance(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;

    let name_ptr = unsafe { llama_cpp_bindings_sys::llama_sampler_name(sampler.as_ptr()) };
    ensure!(!name_ptr.is_null());
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?;

    ensure!(name == "llguidance");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_sampler_cloned_through_ffi_keeps_its_own_grammar_state(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;
    let mut cloned = unsafe {
        LlamaSampler::from_raw(
            llama_cpp_bindings_sys::llama_sampler_clone(sampler.as_ptr()),
            "llguidance",
        )
    }?;

    ensure!(cloned.accept(single_token(model, "yes")?) == Ok(()));
    ensure!(sampler.accept(single_token(model, "no")?) == Ok(()));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn reset_rolls_back_accepted_tokens(&LlamaFixture { model, .. }: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;

    let yes_tokens = model.str_to_token("yes", AddBos::Never, ParseSpecialTokens::Always)?;
    let first_allowed_token = *yes_tokens
        .first()
        .context("the tokenizer must produce a token for \"yes\"")?;

    sampler
        .accept(first_allowed_token)
        .context("the grammar must accept the first token of an allowed word")?;

    sampler
        .reset()
        .context("reset must roll a healthy grammar matcher back to its initial state")?;

    ensure!(
        sampler.accept(first_allowed_token) == Ok(()),
        "after reset the same token must be acceptable again, which is what rollback means"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn accept_maps_an_out_of_vocabulary_token_to_grammar_callback_failure(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;

    ensure!(
        sampler.accept(LlamaToken(i32::MAX - 1))
            == Err(SamplerError::AcceptGrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn sampling_through_a_grammar_in_an_error_state_reports_the_grammar_failure(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut context = fixture.build_context()?;
    let tokens = model.str_to_token(
        "Answer yes or no:",
        AddBos::Always,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let mut chain = LlamaSampler::chain([
        LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?,
        LlamaSampler::greedy()?,
    ])?;

    ensure!(
        chain.accept(LlamaToken(i32::MAX - 1))
            == Err(SamplerError::AcceptGrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
    );
    ensure!(
        chain.sample(
            &context,
            batch
                .n_tokens()
                .checked_sub(1)
                .context("`batch.n_tokens() - 1` must not overflow")?
        ) == Err(SamplerError::SampleGrammarCallbackFailed {
            message: out_of_vocabulary_failure(),
        })
    );

    Ok(())
}
