use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::chat_assistant_turn::ChatAssistantTurn;
use llama_cpp_bindings::chat_session::ChatSession;
use llama_cpp_bindings::chat_session_options::ChatSessionOptions;
use llama_cpp_bindings::chat_tool_choice::ChatToolChoice;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::user_message::user_message;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
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
fn raw_prompt_completion_usage_matches_the_classified_tokens(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut ctx = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )
    .with_context(|| "unable to create context")?;

    let prompt = "Hello my name is";
    let max_generated_tokens: i32 = 64;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)
        .context(format!("failed to tokenize {prompt}"))?;
    let prompt_token_count = u64::try_from(tokens_list.len())?;

    let mut batch = LlamaBatch::new(ctx.limits());
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens_list, 0, false)?;

    ctx.decode(&batch)
        .with_context(|| "llama_decode() failed")?;

    classifier.commit_prompt_tokens();

    let mut sampler = LlamaSampler::chain([LlamaSampler::dist(1234)?, LlamaSampler::greedy()?])?;
    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut ctx,
        batch: &mut batch,
        initial_position,
        max_generated_tokens,
    }
    .run()?;
    let total_observed =
        outcome.observed_content.checked_add(outcome.observed_reasoning).context("`outcome.observed_content + outcome.observed_reasoning` must not overflow")?.checked_add(outcome.observed_undeterminable).context("`outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable` must not overflow")?;

    ensure!(
        !outcome.generated.text().is_empty(),
        "model should generate at least one token"
    );
    ensure!(
        outcome.observed_tool_call == 0,
        "raw prompt without tool-call markers must not produce ToolCall tokens; \
         outcome={outcome:?}"
    );
    ensure!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();
    ensure!(
        usage.prompt == prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    ensure!(
        usage.content == outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    ensure!(
        usage.reasoning == outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    ensure!(
        usage.undeterminable == outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    ensure!(
        usage.tool_call == outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );
    ensure!(
        usage.completion_tokens() == total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 512,
    n_ubatch = 128,
)]
fn chat_completion_usage_matches_the_classified_tokens(
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

    let messages = [user_message("Hello! How are you?")];
    let session = ChatSession::new(
        model,
        model.chat_templates()?,
        &messages,
        &[],
        &ChatSessionOptions {
            assistant_turn: ChatAssistantTurn::New,
            enable_thinking: true,
            parallel_tool_calls: false,
            tool_choice: ChatToolChoice::Auto,
        },
    )?;

    let tokens = session.tokenize_prompt(model)?;
    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    let prompt_token_count = u64::try_from(tokens.len())?;

    let mut batch = LlamaBatch::new(context.limits());
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&batch)?;

    classifier.commit_prompt_tokens();

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
    .run()?;

    ensure!(
        !outcome.generated.text().is_empty(),
        "model should generate at least one token"
    );
    let total_observed =
        outcome.observed_content.checked_add(outcome.observed_reasoning).context("`outcome.observed_content + outcome.observed_reasoning` must not overflow")?.checked_add(outcome.observed_undeterminable).context("`outcome.observed_content + outcome.observed_reasoning + outcome.observed_undeterminable` must not overflow")?;
    ensure!(
        total_observed > 0,
        "model must produce at least one classified token; outcome={outcome:?}"
    );
    ensure!(
        outcome.observed_tool_call == 0,
        "chat without tool definitions must not produce ToolCall tokens; outcome={outcome:?}"
    );

    let usage = classifier.into_usage();

    ensure!(
        usage.prompt == prompt_token_count,
        "prompt_tokens must equal the tokenizer's prompt length"
    );
    ensure!(
        usage.content == outcome.observed_content,
        "content_tokens must equal observed Content variants"
    );
    ensure!(
        usage.reasoning == outcome.observed_reasoning,
        "reasoning_tokens must equal observed Reasoning variants"
    );
    ensure!(
        usage.undeterminable == outcome.observed_undeterminable,
        "undeterminable_tokens must equal observed Undeterminable variants"
    );
    ensure!(
        usage.completion_tokens() == total_observed,
        "completion_tokens must equal Content + Reasoning + Undeterminable"
    );
    ensure!(
        usage.tool_call == outcome.observed_tool_call,
        "tool_call_tokens must equal observed ToolCall variants"
    );

    Ok(())
}
