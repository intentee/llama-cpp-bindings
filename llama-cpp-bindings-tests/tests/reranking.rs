use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings_tests::normalize::normalize;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

fn cosine_similarity(vec_a: &[f32], vec_b: &[f32]) -> f32 {
    vec_a
        .iter()
        .zip(vec_b.iter())
        .map(|(left, right)| left * right)
        .sum::<f32>()
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 2048,
    n_ubatch = 512,
    n_seq_max = 2,
    n_threads_batch = 8,
    embeddings = true,
)]
fn reranking_produces_scores(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;

    let query = "What is machine learning?";
    let documents = [
        "Machine learning is a subset of artificial intelligence.",
        "The weather today is sunny and warm.",
    ];

    let document_count = documents.len();
    assert_eq!(
        u32::try_from(document_count)?,
        fixture.context_params.n_seq_max,
        "attribute n_seq_max must match the document count this trial expects",
    );

    let mut ctx = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt_lines: Vec<String> = documents
        .iter()
        .map(|document| format!("{query}</s><s>{document}"))
        .collect();

    let tokens_lines_list = prompt_lines
        .iter()
        .map(|line| model.str_to_token(line, AddBos::Always, ParseSpecialTokens::Always))
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| "failed to tokenize prompts")?;

    let n_ctx = usize::try_from(ctx.n_ctx())?;

    if tokens_lines_list.iter().any(|tokens| n_ctx < tokens.len()) {
        bail!("one of the provided prompts exceeds the size of the context window");
    }

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut batch = LlamaBatch::new(ContextLimits::of(&ctx));

    for (sequence_index, tokens) in tokens_lines_list.iter().enumerate() {
        classifier.feed_prompt_sequence_to_batch(
            &mut batch,
            tokens,
            i32::try_from(sequence_index)?,
            false,
        )?;
    }

    let total_tokens: usize = tokens_lines_list.iter().map(Vec::len).sum();
    let total_token_count = u64::try_from(total_tokens)?;

    assert_eq!(classifier.pending_prompt_tokens(), total_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    ctx.memory_clear();
    ctx.decode(&batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, total_token_count);

    let mut embeddings = Vec::with_capacity(document_count);

    for sequence_index in 0..document_count {
        let raw_embedding = ctx
            .embeddings_seq_ith(i32::try_from(sequence_index)?)
            .with_context(|| "failed to get sequence embeddings")?;
        embeddings.push(normalize(raw_embedding));
    }

    assert_eq!(
        embeddings.len(),
        document_count,
        "should produce one embedding per document"
    );

    for (index, embedding) in embeddings.iter().enumerate() {
        assert!(
            !embedding.is_empty(),
            "embedding {index} should not be empty"
        );
    }

    let similarity = cosine_similarity(&embeddings[0], &embeddings[1]);

    assert!(
        similarity.is_finite(),
        "cosine similarity should be a finite number"
    );

    let usage = classifier.into_usage();
    assert_eq!(usage.prompt_tokens, total_token_count);
    assert_eq!(usage.completion_tokens(), 0);

    Ok(())
}
