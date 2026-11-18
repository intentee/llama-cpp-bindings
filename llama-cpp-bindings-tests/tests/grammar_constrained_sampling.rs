use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_bindings_tests::logits_left_by::logits_left_by;
use llama_cpp_bindings_tests::single_token::single_token;
use llama_cpp_context::llama_batch::LlamaBatch;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::add_bos::AddBos;
use llama_cpp_model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_model::sampled_token::SampledToken;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const YES_TOKEN_AND_END_OF_GENERATION: u64 = 2;

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
fn grammar_sampler_constrains_the_first_token_of_the_answer_to_yes(
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

    let prompt = "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(context.limits());

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::grammar(model, r"root ::= [Yy] [Ee] [Ss] | [Nn] [Oo]", "root")?,
        LlamaSampler::temp(0.8)?,
        LlamaSampler::greedy()?,
    ])?;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut outcomes = Vec::new();
    let turn = classifier.sample(
        &mut sampler,
        &context,
        batch
            .n_tokens()
            .checked_sub(1)
            .context("`batch.n_tokens() - 1` must not overflow")?,
        &mut outcomes,
    )?;
    classifier.finish(&mut outcomes)?;

    ensure!(
        outcomes.len() == 1,
        "expected one finalised outcome after finishing"
    );
    let outcome = outcomes.first().context("outcomes has no element 0")?;

    let raw_as_sampled = SampledToken::Content(turn.token);
    ensure!(
        !model.is_eog_token(&raw_as_sampled),
        "Grammar sampler should not allow EOS as first token"
    );

    let piece = outcome.piece.raw();

    ensure!(piece == "Yes");

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
fn grammar_constrained_chat_answer_is_yes_content_that_ends_generation(
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

    let session = closed_reasoning_chat_session(model, "Is the sky blue? yes or no", &[])?;
    let tokens = session.tokenize_prompt(model)?;
    let mut batch = LlamaBatch::new(context.limits());

    let mut classifier = SampledTokenClassifier::for_chat_session(model, session)?;
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&batch)?;
    classifier.commit_prompt_tokens();

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::grammar(model, r#"root ::= "yes" | "no""#, "root")?,
        LlamaSampler::temp(0.8)?,
        LlamaSampler::greedy()?,
    ])?;

    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        classifier: &mut classifier,
        sampler: &mut sampler,
        context: &mut context,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: 10,
    }
    .run()?;

    ensure!(outcome.generated.text() == "yes");
    ensure!(
        outcome.eog_seen,
        "loop must terminate via EOG once grammar accepts, not by exhausting the budget; outcome={outcome:?}"
    );
    ensure!(outcome.observed_reasoning == 0);
    ensure!(outcome.observed_undeterminable == 0);
    ensure!(outcome.observed_tool_call == 0);
    ensure!(outcome.observed_content == YES_TOKEN_AND_END_OF_GENERATION);

    let usage = classifier.into_usage();
    ensure!(usage.completion_tokens() == outcome.observed_content);
    ensure!(usage.reasoning == 0);
    ensure!(usage.undeterminable == 0);

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
fn a_grammar_sampler_masks_tokens_that_cannot_start_the_grammar(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let sampler = LlamaSampler::grammar(model, "root ::= \"hello\"", "root")?;

    ensure!(
        logits_left_by(
            &sampler,
            &[single_token(model, "hello")?, single_token(model, "yes")?]
        )? == [0.0, f32::NEG_INFINITY]
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_with_a_token_reference_constrains_the_first_token(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let think_token = single_token(model, "<think>")?;
    let mut context = LlamaContext::from_model(
        model,
        backend,
        (*context_params).into_llama_context_params(),
    )?;
    let prompt_tokens = model.str_to_token(
        "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
        AddBos::Never,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(context.limits());

    batch.add_sequence(&prompt_tokens, 0, false)?;
    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::grammar(model, r#"root ::= <think> "x""#, "root")?,
        LlamaSampler::greedy()?,
    ])?;

    ensure!(
        sampler.sample(
            &context,
            batch
                .n_tokens()
                .checked_sub(1)
                .context("`batch.n_tokens() - 1` must not overflow")?
        )? == think_token
    );

    Ok(())
}
