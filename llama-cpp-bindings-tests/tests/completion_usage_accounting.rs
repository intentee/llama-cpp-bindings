use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::BareJsonToolCalls;
use llama_cpp_bindings::ChatAssistantTurn;
use llama_cpp_bindings::ChatSession;
use llama_cpp_bindings::ChatSessionOptions;
use llama_cpp_bindings::ChatToolChoice;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::LlamaChatMessage;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

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
fn raw_prompt_completion_usage_matches_the_classified_tokens(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut ctx = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let max_generated_tokens: i32 = 64;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)
        .context(format!("failed to tokenize {prompt}"))?;
    let prompt_token_count = u64::try_from(tokens_list.len())?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&ctx));
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens_list, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    ctx.decode(&batch)
        .with_context(|| "llama_decode() failed")?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, prompt_token_count);

    let mut sampler =
        LlamaSampler::chain_simple([LlamaSampler::dist(1234)?, LlamaSampler::greedy()?])?;
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut ctx,
        batch: &mut batch,
        initial_position,
        max_generated_tokens,
    }
    .run();
    let total_observed =
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable;

    assert!(
        !outcome.generated.text().is_empty(),
        "model should generate at least one token"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "raw prompt without tool-call markers must not produce ToolCall tokens; \
         outcome={outcome:?}"
    );
    assert!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();
    assert_eq!(
        usage.prompt_tokens, prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens, outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens, outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.undeterminable_tokens, outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    assert_eq!(
        usage.tool_call_tokens, outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn chat_completion_usage_matches_the_classified_tokens(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let messages = [LlamaChatMessage::new(
        "user".to_string(),
        "Hello! How are you?".to_string(),
    )?];
    let session = ChatSession::new(
        model,
        &messages,
        "[]",
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    let tokens = session.tokenize_prompt()?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    assert_eq!(classifier.pending_prompt_tokens(), prompt_token_count);
    assert_eq!(classifier.usage().prompt_tokens, 0);

    context.decode(&batch)?;

    let promoted = classifier.commit_prompt_tokens();
    assert_eq!(promoted, prompt_token_count);

    let mut sampler = LlamaSampler::greedy()?;
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 512,
    }
    .run();

    assert!(
        !outcome.generated.text().is_empty(),
        "model should generate at least one token"
    );
    let total_observed =
        outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable;
    assert!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.observed_tool_call, 0,
        "chat without tool definitions must not produce ToolCall tokens; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();

    assert_eq!(
        usage.prompt_tokens, prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    assert_eq!(
        usage.content_tokens, outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    assert_eq!(
        usage.reasoning_tokens, outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    assert_eq!(
        usage.undeterminable_tokens, outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    assert_eq!(
        usage.completion_tokens(),
        total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );
    assert_eq!(
        usage.tool_call_tokens, outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );

    Ok(())
}
