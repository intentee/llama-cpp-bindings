use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

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
