use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::generation_progress::GenerationProgress;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn sample_returns_result_and_succeeds_with_valid_index(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let tokens = model.str_to_token("Hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain([LlamaSampler::temp(0.8)?, LlamaSampler::greedy()?])?;

    let result = sampler.sample(&context, batch.n_tokens() - 1);

    assert!(result.is_ok());
    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn sample_without_grammar_produces_multiple_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt =
        "<|im_start|>user\nSay hello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain([LlamaSampler::temp(0.8)?, LlamaSampler::greedy()?])?;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut outcomes = Vec::new();
    let mut sampled_count: u64 = 0;

    for (position, _) in (batch.n_tokens()..).zip(0..5) {
        let turn = classifier.sample(&mut sampler, &context, -1, &mut outcomes)?;

        if turn.progress == GenerationProgress::Ended {
            break;
        }

        sampled_count += 1;

        let raw_as_sampled = SampledToken::Content(turn.token);

        batch.clear();
        batch.add(&raw_as_sampled, position, &[0], true)?;

        context.decode(&batch)?;
    }

    classifier.finish(&mut outcomes)?;

    assert!(
        sampled_count > 0,
        "Should produce at least one token without grammar"
    );
    let usage = classifier.into_usage();
    assert!(
        usage.completion_tokens() >= sampled_count,
        "completion_tokens ({}) must include the {sampled_count} non-EOG samples",
        usage.completion_tokens()
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn apply_runs_sampler_over_token_data_array(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("Hi", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let mut data_array = context.token_data_array_ith(batch.n_tokens() - 1)?;
    let sampler = LlamaSampler::greedy()?;
    sampler.apply(&mut data_array)?;

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
)]
fn sample_returns_token_after_decode(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("Hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;
    let mut sampler = LlamaSampler::chain([LlamaSampler::temp(0.8)?, LlamaSampler::greedy()?])?;
    let result = sampler.sample(&context, batch.n_tokens() - 1);

    assert!(result.is_ok());

    Ok(())
}
