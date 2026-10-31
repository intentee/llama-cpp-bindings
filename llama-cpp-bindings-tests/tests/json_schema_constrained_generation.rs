use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::json_schema_to_grammar::json_schema_to_grammar;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

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
fn json_schema_grammar_sampler_constrains_output_to_json(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let prompt = "<|im_start|>user\nWhat is 2+2? Respond with a JSON object.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    let tokens = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));

    batch.add_sequence(&tokens, 0, false)?;

    context.decode(&batch)?;

    let grammar_str = json_schema_to_grammar(
        r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#,
    )?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::grammar(model, &grammar_str, "root")?,
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

    assert!(
        piece.starts_with('{'),
        "JSON schema grammar should constrain first token to start with '{{', got: '{piece}'"
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
fn json_schema_constrains_output(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model = fixture.model;
    let backend = fixture.backend;

    let prompt = "The weather in Paris is sunny and 22 degrees. Extract as JSON:\n";

    let mut ctx = LlamaContext::from_model(
        model,
        backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let tokens_list = model.str_to_token(prompt, AddBos::Always, ParseSpecialTokens::Always)?;

    let mut batch = LlamaBatch::new(ContextLimits::of(&ctx));
    let last_index = i32::try_from(tokens_list.len())? - 1;

    for (index, token) in (0_i32..).zip(&tokens_list) {
        batch.add(
            &SampledToken::Content(*token),
            index,
            &[0],
            index == last_index,
        )?;
    }

    ctx.decode(&batch)?;

    let schema = r#"{
  "type": "object",
  "properties": {
    "city": { "type": "string" },
    "temperature": { "type": "number" }
  },
  "required": ["city", "temperature"]
}"#;

    let llg_sampler = LlamaSampler::llguidance(model, "json", schema)?;
    let mut sampler = LlamaSampler::chain([llg_sampler, LlamaSampler::greedy()?])?;

    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut generated = String::new();

    while n_cur <= 128 {
        let token = SampledToken::Content(sampler.sample(&ctx, batch.n_tokens() - 1)?);

        if model.is_eog_token(&token) {
            break;
        }

        let output_string = model.token_to_piece(&token, &mut decoder, true, None)?;
        generated.push_str(&output_string);

        batch.clear();
        batch.add(&token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&batch)?;
    }

    let parsed = serde_json::Deserializer::from_str(&generated)
        .into_iter::<serde_json::Value>()
        .next()
        .context("model produced no JSON value")??;

    assert!(parsed.get("city").is_some());
    assert!(parsed.get("temperature").is_some());

    Ok(())
}
