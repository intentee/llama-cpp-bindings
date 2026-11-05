use std::ffi::CStr;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::error::grammar_error::GrammarError;
use llama_cpp_bindings::error::grammar_runtime_error::GrammarRuntimeError;
use llama_cpp_bindings::error::sample_error::SampleError;
use llama_cpp_bindings::error::sampler_accept_error::SamplerAcceptError;
use llama_cpp_bindings::error::token_to_string_error::TokenToStringError;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::llguidance_sampler::create_llg_sampler;
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

fn yes_no_and_hello(model: &LlamaModel) -> [LlamaToken; 3] {
    [
        single_token(model, "yes"),
        single_token(model, "no"),
        single_token(model, "hello"),
    ]
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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let sampler = create_llg_sampler(model, "json", JSON_SCHEMA)?;

    assert_eq!(
        logits_left_by(
            &sampler,
            &[single_token(model, "{"), single_token(model, "hello")]
        ),
        [0.0, f32::NEG_INFINITY]
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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let sampler = LlamaSampler::llguidance(model, "regex", REGEX_GRAMMAR)?;

    assert_eq!(
        logits_left_by(&sampler, &yes_no_and_hello(model)),
        [0.0, 0.0, f32::NEG_INFINITY]
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
fn a_lark_sampler_masks_tokens_that_cannot_start_the_grammar(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let sampler = create_llg_sampler(model, "lark", LARK_GRAMMAR)?;

    assert_eq!(
        logits_left_by(&sampler, &yes_no_and_hello(model)),
        [0.0, 0.0, f32::NEG_INFINITY]
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
fn an_unknown_grammar_kind_is_reported_as_an_invalid_grammar(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        create_llg_sampler(fixture.model, "not_a_real_kind", "anything").err(),
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
fn a_json_schema_that_is_not_json_is_reported_as_an_invalid_grammar(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        create_llg_sampler(fixture.model, "json", "{this is not valid json").err(),
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
fn a_malformed_regex_leaves_llguidance_without_a_parser(fixture: &LlamaFixture<'_>) -> Result<()> {
    let Err(GrammarError::LlguidanceParserUnavailable { .. }) =
        create_llg_sampler(fixture.model, "regex", "[invalid")
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
fn name_callback_returns_llguidance(fixture: &LlamaFixture<'_>) -> Result<()> {
    let sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    let name_ptr = unsafe { llama_cpp_bindings_sys::llama_sampler_name(sampler.as_ptr()) };
    assert!(!name_ptr.is_null());
    let name = unsafe { CStr::from_ptr(name_ptr) }.to_str()?;

    assert_eq!(name, "llguidance");

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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut cloned = unsafe {
        LlamaSampler::from_raw(
            llama_cpp_bindings_sys::llama_sampler_clone(sampler.as_ptr()),
            "llguidance",
        )
    }?;

    assert_eq!(cloned.accept(single_token(model, "yes")), Ok(()));
    assert_eq!(sampler.accept(single_token(model, "no")), Ok(()));

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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "Answer yes or no:";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let llg_sampler = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let mut chain = LlamaSampler::chain([llg_sampler, LlamaSampler::greedy()?])?;

    let token = chain.sample(&context, batch.n_tokens() - 1)?;
    let piece = model.token_to_piece(
        &SampledToken::Content(token),
        &mut encoding_rs::UTF_8.new_decoder(),
        false,
        None,
    )?;

    assert_eq!(piece, "yes");

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
fn reset_rolls_back_accepted_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    let yes_tokens =
        fixture
            .model
            .str_to_token("yes", AddBos::Never, ParseSpecialTokens::Always)?;
    let first_allowed_token = *yes_tokens
        .first()
        .context("the tokenizer must produce a token for \"yes\"")?;

    sampler
        .accept(first_allowed_token)
        .context("the grammar must accept the first token of an allowed word")?;

    sampler
        .reset()
        .context("reset must roll a healthy grammar matcher back to its initial state")?;

    assert_eq!(
        sampler.accept(first_allowed_token),
        Ok(()),
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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut sampler = create_llg_sampler(fixture.model, "regex", REGEX_GRAMMAR)?;

    assert_eq!(
        sampler.accept(LlamaToken(i32::MAX - 1)),
        Err(SamplerAcceptError::GrammarCallbackFailed {
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
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let candidates = yes_no_and_hello(model);
    let first = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;
    let second = create_llg_sampler(model, "regex", REGEX_GRAMMAR)?;

    assert_eq!(
        logits_left_by(&first, &candidates),
        logits_left_by(&second, &candidates)
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
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let mut chain = LlamaSampler::chain([
        create_llg_sampler(model, "regex", REGEX_GRAMMAR)?,
        LlamaSampler::greedy()?,
    ])?;

    assert_eq!(
        chain.accept(LlamaToken(i32::MAX - 1)),
        Err(SamplerAcceptError::GrammarCallbackFailed {
            message: out_of_vocabulary_failure(),
        })
    );
    assert_eq!(
        chain.sample(&context, batch.n_tokens() - 1),
        Err(SampleError::GrammarCallbackFailed {
            message: out_of_vocabulary_failure(),
        })
    );

    Ok(())
}

fn freshly_loaded(fixture: &LlamaFixture<'_>) -> Result<LlamaModel> {
    Ok(LlamaModel::load_from_file(
        fixture.backend,
        fixture.model_path,
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

    assert_eq!(
        create_llg_sampler(&model, "regex", REGEX_GRAMMAR).err(),
        Some(GrammarError::TokEnvUnavailable(
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

    assert_eq!(
        model.approximate_tok_env().err(),
        Some(TokenToStringError::InsufficientBufferSpace(-34))
    );

    Ok(())
}
