use std::ffi::CString;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use llama_cpp_bindings::BareJsonToolCalls;
use llama_cpp_bindings::GbnfValidationError;
use llama_cpp_bindings::GrammarError;
use llama_cpp_bindings::SampledToken;
use llama_cpp_bindings::SampledTokenClassifier;
use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::AddBos;
use llama_cpp_bindings::model::ParseSpecialTokens;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_fault_injection::ArmedFault;
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
fn grammar_sampler_constrains_output_to_yes_or_no(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "<|im_start|>user\nIs the sky blue? Answer yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, r"root ::= [Yy] [Ee] [Ss] | [Nn] [Oo]", "root")?,
        LlamaSampler::temp(0.8)?,
        LlamaSampler::greedy()?,
    ])?;

    let mut classifier = SampledTokenClassifier::undeterminable(model);
    let mut outcomes = Vec::new();
    let turn = classifier.sample(&mut sampler, &context, batch.n_tokens() - 1, &mut outcomes)?;
    classifier.finish(&mut outcomes)?;

    assert_eq!(
        outcomes.len(),
        1,
        "expected one finalised outcome after finishing"
    );
    let outcome = &outcomes[0];

    let raw_as_sampled = SampledToken::Content(turn.token);
    assert!(
        !model.is_eog_token(&raw_as_sampled),
        "Grammar sampler should not allow EOS as first token"
    );

    let piece = outcome.piece.raw();
    let first_char = piece
        .chars()
        .next()
        .ok_or_else(|| anyhow!("piece should have at least one character"))?
        .to_lowercase()
        .next()
        .ok_or_else(|| anyhow!("lowercase iterator should yield a character"))?;

    assert!(
        first_char == 'y' || first_char == 'n',
        "Grammar should constrain first token to start with y/n, got: '{piece}'"
    );
    assert_eq!(
        classifier.usage().completion_tokens(),
        1,
        "exactly one completion token sampled"
    );

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
fn sample_with_grammar_produces_constrained_output_in_loop(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let session = closed_reasoning_chat_session(model, "Is the sky blue? yes or no", "[]");
    let tokens = session.tokenize_prompt()?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    let mut classifier =
        SampledTokenClassifier::for_chat_session(&session, BareJsonToolCalls::Detect);
    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;

    context.decode(&batch)?;
    classifier.commit_prompt_tokens();

    let mut sampler = LlamaSampler::chain_simple([
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
    .run();

    let lowercase = outcome.generated.text().to_lowercase();
    assert!(
        lowercase == "yes" || lowercase == "no",
        "Grammar loop should produce 'yes' or 'no', got: '{}'",
        outcome.generated.text()
    );
    assert!(
        outcome.eog_seen,
        "loop must terminate via EOG once grammar accepts, not by exhausting the budget; outcome={outcome:?}"
    );
    assert_eq!(outcome.observed_reasoning, 0);
    assert_eq!(outcome.observed_undeterminable, 0);
    assert_eq!(outcome.observed_tool_call, 0);
    assert!(outcome.observed_content > 0);

    let usage = classifier.into_usage();
    assert_eq!(usage.completion_tokens(), outcome.observed_content);
    assert_eq!(usage.reasoning_tokens, 0);
    assert_eq!(usage.undeterminable_tokens, 0);

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
fn grammar_returns_sampler_for_valid_grammar(fixture: &LlamaFixture<'_>) {
    let sampler = LlamaSampler::grammar(fixture.model, "root ::= \"hello\"", "root");

    assert!(sampler.is_ok());
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_with_a_token_reference_constrains_the_first_token(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = fixture.model;
    let [think_token] =
        model.str_to_token("<think>", AddBos::Never, ParseSpecialTokens::Always)?[..]
    else {
        bail!("<think> must be a single token");
    };
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;
    let prompt_tokens = model.str_to_token(
        "<|im_start|>user\nSay hi<|im_end|>\n<|im_start|>assistant\n",
        AddBos::Never,
        ParseSpecialTokens::Always,
    )?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&prompt_tokens, 0, false)?;
    context.decode(&batch)?;

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(model, r#"root ::= <think> "x""#, "root")?,
        LlamaSampler::greedy()?,
    ])?;

    assert_eq!(sampler.sample(&context, batch.n_tokens() - 1)?, think_token);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_without_its_root_reports_the_missing_root(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaSampler::grammar(fixture.model, r#"answer ::= <think> "x""#, "root").err(),
        Some(GrammarError::RootNotFound)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 256,
    n_batch = 128,
    n_ubatch = 64,
)]
fn qwen35_grammar_with_a_syntax_error_reports_it(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaSampler::grammar(fixture.model, "root ::= (<think>", "root").err(),
        Some(GrammarError::GrammarRejected(
            GbnfValidationError::SyntaxError
        ))
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
fn a_grammar_with_a_nul_byte_is_refused(fixture: &LlamaFixture<'_>) {
    assert_eq!(
        LlamaSampler::grammar(fixture.model, "root ::= \"a\0\"", "root").err(),
        CString::new("root ::= \"a\0\"")
            .err()
            .map(GrammarError::GrammarContainsNul)
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn a_valid_grammar_llama_cpp_refuses_to_load_is_reported_malformed(fixture: &LlamaFixture<'_>) {
    let _fault = ArmedFault::at(
        llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SAMPLER_INIT_GRAMMAR_RETURNS_NULL,
    );

    assert_eq!(
        LlamaSampler::grammar(fixture.model, "root ::= \"a\"", "root").err(),
        Some(GrammarError::GrammarMalformed)
    );
}
