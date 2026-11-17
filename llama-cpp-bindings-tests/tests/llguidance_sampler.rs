use std::ffi::CStr;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::error::grammar_runtime_error::GrammarRuntimeError;
use llama_cpp_bindings::error::sample_error::SampleError;
use llama_cpp_bindings::error::sampler_accept_error::SamplerAcceptError;
use llama_cpp_bindings::error::token_to_string_error::TokenToStringError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::LlamaModel;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const JSON_SCHEMA: &str =
    r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;
const REGEX_GRAMMAR: &str = r"yes|no";

const LARK_GRAMMAR: &str = r#"start: "yes" | "no""#;

fn out_of_vocabulary_failure() -> String {
    GrammarRuntimeError::InternalParserError {
        message: "token id 2147483646 out of range".to_owned(),
    }
    .to_string()
}

fn yes_no_and_hello(model: &LlamaModel) -> Result<[LlamaToken; 3]> {
    Ok([
        single_token(model, "yes")?,
        single_token(model, "no")?,
        single_token(model, "hello")?,
    ])
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
fn a_json_schema_sampler_masks_tokens_that_cannot_start_the_schema(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::llguidance(model, "json", JSON_SCHEMA)?;

    ensure!(
        logits_left_by(
            &sampler,
            &[single_token(model, "{")?, single_token(model, "hello")?]
        )? == [0.0, f32::NEG_INFINITY]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
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
fn a_regex_sampler_masks_tokens_that_cannot_start_a_match(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

    ensure!(logits_left_by(&sampler, &yes_no_and_hello(model)?)? == [0.0, 0.0, f32::NEG_INFINITY]);

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
fn a_lark_sampler_masks_tokens_that_cannot_start_the_grammar(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::llguidance(model, "lark", LARK_GRAMMAR)?;

    ensure!(logits_left_by(&sampler, &yes_no_and_hello(model)?)? == [0.0, 0.0, f32::NEG_INFINITY]);

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
fn an_unknown_grammar_kind_is_reported_as_an_invalid_grammar(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        LlamaSampler::llguidance(model, "not_a_real_kind", "anything").err(),
        Some(GrammarError::LlguidanceGrammarInvalid {
            message: "unknown constraint type: not_a_real_kind".to_owned(),
        })
    );
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
fn a_json_schema_that_is_not_json_is_reported_as_an_invalid_grammar(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        LlamaSampler::llguidance(model, "json", "{this is not valid json").err(),
        Some(GrammarError::LlguidanceGrammarInvalid {
            message: "key must be a string at line 1 column 2".to_owned(),
        })
    );
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
fn a_malformed_regex_leaves_llguidance_without_a_parser(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let Err(GrammarError::LlguidanceParserUnavailable { .. }) =
        LlamaSampler::llguidance(model, "regex", "[invalid")
    else {
        bail!("a regex with an unclosed character class must leave llguidance without a parser");
    };

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
fn name_callback_returns_llguidance(&LlamaFixture { model, .. }: &LlamaFixture<'_>) -> Result<()> {
    let sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

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
    let mut sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;
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
fn llguidance_chain_samples_yes_as_the_first_answer_token(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;

    let prompt = "Answer yes or no:";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let llg_sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain([llg_sampler, LlamaSampler::greedy()?])?;

    let token = chain.sample(
        &context,
        batch
            .n_tokens()
            .checked_sub(1)
            .context("`batch.n_tokens() - 1` must not overflow")?,
    )?;
    let piece = model.token_to_piece(
        SampledToken::Content(token),
        &mut encoding_rs::UTF_8.new_decoder(),
        false,
        None,
    )?;

    ensure!(piece == "yes");

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
    let mut sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

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
    let mut sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

    ensure!(
        sampler.accept(LlamaToken(i32::MAX - 1))
            == Err(SamplerAcceptError::GrammarCallbackFailed {
                message: out_of_vocabulary_failure(),
            })
    );

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
fn samplers_sharing_the_cached_tokenizer_environment_mask_alike(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let candidates = yes_no_and_hello(model)?;
    let first = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;
    let second = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

    ensure!(logits_left_by(&first, &candidates)? == logits_left_by(&second, &candidates)?);

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
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let mut chain = LlamaSampler::chain([
        LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?,
        LlamaSampler::greedy()?,
    ])?;

    ensure!(
        chain.accept(LlamaToken(i32::MAX - 1))
            == Err(SamplerAcceptError::GrammarCallbackFailed {
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
        ) == Err(SampleError::GrammarCallbackFailed {
            message: out_of_vocabulary_failure(),
        })
    );

    Ok(())
}

fn freshly_loaded(
    &LlamaFixture {
        backend,
        model_path,
        ..
    }: &LlamaFixture<'_>,
) -> Result<LlamaModel> {
    Ok(LlamaModel::load_from_file(
        backend,
        model_path,
        &LlamaModelParams::default(),
    )?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn creating_a_sampler_reports_a_vocabulary_llama_cpp_cannot_render(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = freshly_loaded(fixture)?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_TOKEN_TO_PIECE_FAILS);

    ensure!(
        LlamaSampler::llguidance(&model, "regex", REGEX_GRAMMAR).err()
            == Some(GrammarError::TokEnvUnavailable(
                TokenToStringError::InsufficientBufferSpace(-34)
            ))
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
fn the_tokenizer_environment_reports_a_control_token_llama_cpp_cannot_render(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = freshly_loaded(fixture)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SPECIAL_TOKEN_TO_PIECE_FAILS);

    ensure!(
        model.approximate_tok_env().err() == Some(TokenToStringError::InsufficientBufferSpace(-34))
    );

    Ok(())
}
