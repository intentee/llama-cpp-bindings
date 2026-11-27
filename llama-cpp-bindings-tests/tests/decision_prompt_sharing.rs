use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;
use indexmap::IndexMap;
use serde_json::Value;
use serde_json::json;

use llama_cpp_bindings_tests::answered_decisions::answered_decisions;
use llama_cpp_bindings_tests::snowy_weather_decision_request::snowy_weather_decision_request;
use llama_cpp_bindings_types::decision_answer::DecisionAnswer;
use llama_cpp_bindings_types::decision_question::DecisionQuestion;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_decision::prepared_decision_request::PreparedDecisionRequest;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn snowy_weather_and_season_request() -> DecisionRequest {
    let mut request = snowy_weather_decision_request();

    request.questions.insert(
        "season".to_owned(),
        DecisionQuestion::Choice {
            criteria: IndexMap::from([
                ("summer".to_owned(), Value::Null),
                ("winter".to_owned(), Value::Null),
            ]),
            instructions: json!("Which season is it?"),
        },
    );

    request
}

fn chose(answers: &IndexMap<String, DecisionAnswer>, question_id: &str, expected: &str) -> bool {
    matches!(
        answers.get(question_id),
        Some(DecisionAnswer::Choice { choice, .. }) if choice == expected
    )
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-BF16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn prompts_sharing_their_leading_tokens_still_answer_their_own_questions(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let request = snowy_weather_and_season_request();
    let prepared = PreparedDecisionRequest::new(&DecisionModel::new(Arc::clone(model))?, &request)?;

    ensure!(prepared.prompts(None)?.shared_prefix_length > 0);

    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &request,
        None,
    )?;

    ensure!(
        chose(&answers, "weather", "snowing") && chose(&answers, "season", "winter"),
        "answers were {answers:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn prompts_of_a_model_that_cannot_share_them_share_no_tokens(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let prepared = PreparedDecisionRequest::new(
        &DecisionModel::new(Arc::clone(model))?,
        &snowy_weather_and_season_request(),
    )?;

    ensure!(prepared.prompts(None)?.shared_prefix_length == 0);

    Ok(())
}
