use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_bindings_tests::yes_or_no_regex::YES_OR_NO_REGEX;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::llama_token::LlamaToken;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const JSON_SCHEMA: &str =
    r#"{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}"#;

const LARK_GRAMMAR: &str = r#"start: "yes" | "no""#;

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
    let sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;

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
    let mut batch = LlamaBatch::new(context.limits());
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let llg_sampler = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;
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
fn samplers_sharing_the_cached_tokenizer_environment_mask_alike(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let candidates = yes_no_and_hello(model)?;
    let first = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;
    let second = LlamaSampler::llguidance(model, "regex", YES_OR_NO_REGEX)?;

    ensure!(logits_left_by(&first, &candidates)? == logits_left_by(&second, &candidates)?);

    Ok(())
}
