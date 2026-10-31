use anyhow::Context;
use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::context::llama_pooling_type::LlamaPoolingType;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::normalize::normalize;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_threads_batch = 8,
    embeddings = true,
)]
fn embedding_generation_produces_vectors(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;

    let mut ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let tokens = model
        .str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)
        .context(format!("failed to tokenize {prompt}"))?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let n_ctx = usize::try_from(ctx.n_ctx())?;
    assert!(tokens.len() <= n_ctx, "prompt exceeds context window size");

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits::of(&ctx));
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    ctx.memory_clear();
    ctx.decode(&batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let embedding = ctx
        .embeddings_seq_ith(0)
        .with_context(|| "failed to get embeddings")?;
    let normalized = normalize(embedding);

    assert!(
        !normalized.is_empty(),
        "embedding should have at least one dimension"
    );

    let magnitude: f32 = normalized
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        (magnitude - 1.0).abs() < 0.01,
        "normalized embedding magnitude should be approximately 1.0, got {magnitude}"
    );

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, prompt_token_count);
    assert_eq!(usage.completion_tokens(), 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn decoding_with_embeddings_enabled_succeeds(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;

    let result = context.decode(&batch);

    assert!(result.is_ok());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_seq_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let embeddings = context.embeddings_seq_ith(0)?;

    assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 4,
    embeddings = true,
)]
fn multi_sequence_embeddings_returns_one_embedding_per_sequence(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let inputs = [
        "alpha is here",
        "beta runs fast",
        "gamma waits",
        "delta jumps",
    ];
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    for (sequence_index, text) in inputs.iter().enumerate() {
        let tokens =
            fixture
                .model
                .str_to_token(text, AddBos::Always, ParseSpecialTokens::Always)?;
        let sequence_id = i32::try_from(sequence_index)?;

        batch.add_sequence(&tokens, sequence_id, true)?;
    }

    context.decode(&batch)?;

    let n_embd = usize::try_from(fixture.model.n_embd())?;
    let mut collected: Vec<Vec<f32>> = Vec::with_capacity(inputs.len());

    for sequence_index in 0..inputs.len() {
        let sequence_id = i32::try_from(sequence_index)?;
        let embedding = context.embeddings_seq_ith(sequence_id)?;

        assert_eq!(
            embedding.len(),
            n_embd,
            "sequence {sequence_index} embedding length mismatch"
        );

        collected.push(embedding.to_vec());
    }

    for (left_index, left) in collected.iter().enumerate() {
        for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
            assert_ne!(
                left, right,
                "embedding for sequence {left_index} must differ from sequence {right_index}",
            );
        }
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 4,
    embeddings = true,
)]
fn embeddings_returns_distinct_values_when_reused_batch_has_extra_capacity(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let iterations = [
        [
            "This is the first document with enough content to contribute meaningfully to the batch size calculation",
            "This is the second document that should be processed in a potentially different batch from the first",
        ],
        [
            "This is the third document adding more content to ensure the total exceeds the configured chunk limit",
            "This is the fourth document which should demonstrate that batching distributes across agent requests",
        ],
    ];

    let n_embd = usize::try_from(fixture.model.n_embd())?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let mut collected: Vec<Vec<f32>> = Vec::new();

    for iteration_inputs in iterations {
        for (sequence_index, text) in iteration_inputs.iter().enumerate() {
            let tokens =
                fixture
                    .model
                    .str_to_token(text, AddBos::Always, ParseSpecialTokens::Always)?;
            let sequence_id = i32::try_from(sequence_index)?;

            batch.add_sequence(&tokens, sequence_id, true)?;
        }

        context.memory_clear();
        context.decode(&batch)?;

        for sequence_index in 0..iteration_inputs.len() {
            let sequence_id = i32::try_from(sequence_index)?;
            let embedding = context.embeddings_seq_ith(sequence_id)?;

            assert_eq!(
                embedding.len(),
                n_embd,
                "iteration sequence {sequence_index} embedding length mismatch"
            );

            collected.push(embedding.to_vec());
        }

        batch.clear();
    }

    assert_eq!(
        collected.len(),
        iterations.iter().flatten().count(),
        "expected one embedding per input across every iteration"
    );

    for (left_index, left) in collected.iter().enumerate() {
        for (right_index, right) in collected.iter().enumerate().skip(left_index + 1) {
            assert_ne!(
                left, right,
                "embedding {left_index} must differ from embedding {right_index} across reused-batch iterations",
            );
        }
    }

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    embeddings = true,
)]
fn embeddings_ith_returns_valid_embeddings(fixture: &LlamaFixture<'_>) -> Result<()> {
    let mut context = LlamaContext::from_model(
        fixture.model,
        fixture.backend,
        (*fixture.context_params)
            .into_llama_context_params()
            .with_pooling_type(LlamaPoolingType::None),
    )?;
    let tokens = fixture
        .model
        .str_to_token("hello", AddBos::Always, ParseSpecialTokens::Always)?;
    let last_index = i32::try_from(tokens.len() - 1)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    batch.add_sequence(&tokens, 0, false)?;
    context.decode(&batch)?;

    let embeddings = context.embeddings_ith(last_index)?;

    assert_eq!(embeddings.len(), usize::try_from(fixture.model.n_embd())?);

    Ok(())
}
