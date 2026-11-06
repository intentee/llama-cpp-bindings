use anyhow::Result;
use serde_json::Value;
use serde_json::json;

use llama_cpp_bindings::context::LlamaContext;
use llama_cpp_bindings::context::context_limits::ContextLimits;
use llama_cpp_bindings::json_schema_to_grammar::json_schema_to_grammar;
use llama_cpp_bindings::llama_batch::LlamaBatch;
use llama_cpp_bindings::model::add_bos::AddBos;
use llama_cpp_bindings::model::parse_special_tokens::ParseSpecialTokens;
use llama_cpp_bindings::sampled_token::SampledToken;
use llama_cpp_bindings::sampled_token_classifier::SampledTokenClassifier;
use llama_cpp_bindings::sampling::LlamaSampler;
use llama_cpp_bindings_tests::classify_sample_loop::ClassifySampleLoop;
use llama_cpp_bindings_tests::closed_reasoning_chat_session::closed_reasoning_chat_session;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

const MAX_ANSWER_TOKENS: i32 = 64;

fn json_schema_constrained_answer(fixture: &LlamaFixture<'_>) -> Result<Value> {
    let model = fixture.model;
    let mut context = LlamaContext::from_model(
        model,
        fixture.backend,
        (*fixture.context_params).into_llama_context_params(),
    )?;

    let session =
        closed_reasoning_chat_session(model, "What is 2+2? Respond with a JSON object.", &[]);
    let tokens = session.tokenize_prompt(fixture.model)?;
    let mut batch = LlamaBatch::new(ContextLimits::of(&context));
    let mut classifier = SampledTokenClassifier::undeterminable(model);

    classifier.feed_prompt_sequence_to_batch(&mut batch, &tokens, 0, false)?;
    context.decode(&batch)?;
    classifier.commit_prompt_tokens();

    let grammar_str = json_schema_to_grammar(
        r#"{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}"#,
    )?;

    let mut sampler = LlamaSampler::chain([
        LlamaSampler::grammar(model, &grammar_str, "root")?,
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
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    assert!(
        outcome.eog_seen,
        "the answer must end once the schema is satisfied; outcome={outcome:?}"
    );

    Ok(outcome.generated.text().parse::<Value>()?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn deepseek_r1_8b_answers_the_json_schema_with_four(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(
        json_schema_constrained_answer(fixture)?,
        json!({"answer": "4"})
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn glm47_flash_answers_the_json_schema_with_a_sentence(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(
        json_schema_constrained_answer(fixture)?,
        json!({"answer": "The sum of 2 and 2 is 4."})
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen35_answers_the_json_schema_with_four(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(
        json_schema_constrained_answer(fixture)?,
        json!({"answer": "4"})
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn qwen36_answers_the_json_schema_with_an_empty_string(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(
        json_schema_constrained_answer(fixture)?,
        json!({"answer": ""})
    );

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
fn an_llguidance_json_schema_constrains_the_answer_to_the_schema(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
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
  "required": ["city", "temperature"],
  "additionalProperties": false
}"#;

    let llg_sampler = LlamaSampler::llguidance(model, "json", schema)?;
    let mut sampler = LlamaSampler::chain([llg_sampler, LlamaSampler::greedy()?])?;

    let initial_position = batch.n_tokens();
    let outcome = ClassifySampleLoop {
        classifier: &mut SampledTokenClassifier::undeterminable(model),
        sampler: &mut sampler,
        context: &mut ctx,
        batch: &mut batch,
        initial_position,
        max_generated_tokens: MAX_ANSWER_TOKENS,
    }
    .run();

    assert!(
        outcome.eog_seen,
        "the answer must end once the schema is satisfied; outcome={outcome:?}"
    );
    assert_eq!(
        outcome.generated.text().parse::<Value>()?,
        json!({"city": "Paris", "temperature": 22})
    );

    Ok(())
}
