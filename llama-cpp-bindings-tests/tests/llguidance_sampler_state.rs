use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_bindings_tests::yes_or_no_regex::YES_OR_NO_REGEX;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

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
            llama_cpp_bindings_sys::bindings::llama_sampler_clone(sampler.as_ptr()),
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
