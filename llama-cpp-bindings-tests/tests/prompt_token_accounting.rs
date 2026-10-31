use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::batch_add_error::BatchAddError;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::token::LlamaToken;
use llama_cpp_bindings_types::TokenUsageError;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn feed_prompt_to_batch_increments_pending_prompt_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 8,
        n_ubatch: 8,
        n_seq_max: 1,
        n_vocab: model.n_vocab(),
    });

    classifier.feed_prompt_to_batch(
        &mut batch,
        model.token_bos().context("model declares a BOS token")?,
        0,
        &[0],
        false,
    )?;
    classifier.feed_prompt_to_batch(
        &mut batch,
        model.token_eos().context("model declares an EOS token")?,
        1,
        &[0],
        false,
    )?;

    assert_eq!(classifier.pending_prompt_tokens(), 2);
    assert_eq!(batch.n_tokens(), 2);

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
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn feed_prompt_sequence_to_batch_stages_all_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 8,
        n_ubatch: 8,
        n_seq_max: 1,
        n_vocab: model.n_vocab(),
    });

    let tokens = vec![
        model.token_bos().context("model declares a BOS token")?,
        model.token_eos().context("model declares an EOS token")?,
        model.token_nl().context("model declares a newline token")?,
    ];
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), 3);
    assert_eq!(batch.n_tokens(), 3);

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
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn commit_prompt_tokens_promotes_pending_count_to_usage_and_clears(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 8,
        n_ubatch: 8,
        n_seq_max: 1,
        n_vocab: model.n_vocab(),
    });

    classifier.feed_prompt_to_batch(
        &mut batch,
        model.token_bos().context("model declares a BOS token")?,
        0,
        &[0],
        false,
    )?;
    classifier.feed_prompt_to_batch(
        &mut batch,
        model.token_eos().context("model declares an EOS token")?,
        1,
        &[0],
        false,
    )?;

    let promoted = classifier.commit_prompt_tokens();

    assert_eq!(promoted, 2);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 2);

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
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn discard_pending_prompt_tokens_clears_count_without_recording_usage(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 8,
        n_ubatch: 8,
        n_seq_max: 1,
        n_vocab: model.n_vocab(),
    });

    classifier.feed_prompt_to_batch(
        &mut batch,
        model.token_bos().context("model declares a BOS token")?,
        0,
        &[0],
        false,
    )?;

    let discarded = classifier.discard_pending_prompt_tokens();

    assert_eq!(discarded, 1);
    assert_eq!(classifier.pending_prompt_tokens(), 0);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn cached_prompt_tokens_cannot_exceed_the_recorded_prompt(fixture: &LlamaFixture<'_>) {
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);

    classifier.record_prompt_tokens(2);

    assert_eq!(
        classifier.record_cached_prompt_tokens(3),
        Err(TokenUsageError::CachedExceedsPrompt {
            cached_after: 3,
            prompt: 2,
        })
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_prompt_token_refused_by_a_full_batch_is_not_counted(fixture: &LlamaFixture<'_>) {
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 0,
        n_ubatch: 0,
        n_seq_max: 1,
        n_vocab: fixture.model.n_vocab(),
    });

    assert_eq!(
        classifier.feed_prompt_to_batch(&mut batch, LlamaToken::new(0), 0, &[0], false),
        Err(BatchAddError::BatchFull { n_batch: 0 })
    );
    assert_eq!(classifier.pending_prompt_tokens(), 0);
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_prompt_sequence_refused_by_a_full_batch_is_not_counted(fixture: &LlamaFixture<'_>) {
    let mut classifier = SampledTokenClassifier::undeterminable(fixture.model);
    let mut batch = LlamaBatch::new(ContextLimits {
        n_batch: 0,
        n_ubatch: 0,
        n_seq_max: 1,
        n_vocab: fixture.model.n_vocab(),
    });

    assert_eq!(
        classifier.feed_prompt_sequence_to_batch(&mut batch, &[LlamaToken::new(0)], 0, false),
        Err(BatchAddError::BatchFull { n_batch: 0 })
    );
    assert_eq!(classifier.pending_prompt_tokens(), 0);
}
