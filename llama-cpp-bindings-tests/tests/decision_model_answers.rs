use std::fs;
use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use hf_hub::Repo;
use indexmap::IndexMap;
use serde_json::Value;
use serde_json::json;

use llama_cpp_bindings_tests::answered_decisions::answered_decisions;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::snowy_weather_and_freezing_decision_request::snowy_weather_and_freezing_decision_request;
use llama_cpp_bindings_tests::snowy_weather_decision_request::snowy_weather_decision_request;
use llama_cpp_bindings_types::decision_answer::DecisionAnswer;
use llama_cpp_bindings_types::decision_question::DecisionQuestion;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_mtmd::mtmd_context::MtmdContext;
use llama_cpp_mtmd::mtmd_context_params::MtmdContextParams;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;
use llama_cpp_test_model_provisioning::download_model::download_model;

fn chose_snowing(answers: &IndexMap<String, DecisionAnswer>) -> bool {
    matches!(
        answers.get("weather"),
        Some(DecisionAnswer::Choice { choice, .. }) if choice == "snowing"
    )
}

#[llama_test(
    model_source = HuggingFace("ggml-org/OpenJev-GGUF", "OpenJev-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_model_answering_with_label_logits_chooses_the_weather_its_state_describes(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_decision_request(),
        None,
    )?;

    ensure!(chose_snowing(&answers), "answers were {answers:?}");

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
fn a_model_answering_at_marker_tokens_chooses_the_weather_its_state_describes(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_decision_request(),
        None,
    )?;

    ensure!(chose_snowing(&answers), "answers were {answers:?}");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-BF16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_model_answering_with_a_pointer_head_chooses_the_weather_its_state_describes(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_decision_request(),
        None,
    )?;

    ensure!(chose_snowing(&answers), "answers were {answers:?}");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("LiquidAI/d1-2B-GGUF", "d1-2B-Q4_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_model_answering_with_grouped_label_logits_chooses_the_weather_its_state_describes(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_decision_request(),
        None,
    )?;

    ensure!(chose_snowing(&answers), "answers were {answers:?}");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_model_answering_every_question_in_one_prompt_answers_each_of_them(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_and_freezing_decision_request(),
        None,
    )?;

    ensure!(chose_snowing(&answers), "answers were {answers:?}");
    ensure!(
        matches!(answers.get("freezing"), Some(DecisionAnswer::Noul { noul }) if *noul > 0.5),
        "answers were {answers:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace(
        "ggml-org-dev/tiny-Decision-2.0-Kai-0.6B-for-testing-GGUF",
        "tiny-Decision-2.0-Kai-0.6B-for-testing-Q4_0.gguf"
    ),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_model_reading_each_question_from_its_own_joint_prompt_answers_each_of_them(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let answers = answered_decisions(
        model,
        backend,
        (*context_params).into_llama_context_params(),
        &snowy_weather_and_freezing_decision_request(),
        None,
    )?;

    ensure!(
        matches!(
            answers.get("weather"),
            Some(DecisionAnswer::Choice { probabilities, .. })
                if probabilities.keys().eq(["sunny", "raining", "snowing"])
        ),
        "answers were {answers:?}"
    );
    ensure!(
        matches!(answers.get("freezing"), Some(DecisionAnswer::Noul { .. })),
        "answers were {answers:?}"
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("openbmb/MiniCPM-OCR-Jev-gguf", "MiniCPM-OCR-Jev-IQ3_XS.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 4096,
    n_batch = 4096,
    n_ubatch = 4096,
)]
fn a_model_answering_with_pooled_class_scores_scores_every_class(
    &LlamaFixture {
        backend,
        context_params,
        model_path,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let model = Arc::new(LlamaModel::load_from_file(
        backend,
        model_path,
        &LlamaModelParams::default().with_n_gpu_layers(-1),
    )?);
    let projector_path = download_model(
        &Repo::model("openbmb/MiniCPM-OCR-Jev-gguf".to_owned()),
        "mmproj-model-f16.gguf",
    )?;
    let projector = MtmdContext::init_from_file(
        projector_path
            .to_str()
            .context("the projector path is valid UTF-8")?,
        Arc::clone(&model),
        &MtmdContextParams {
            media_marker: DecisionModel::media_marker().to_owned(),
            ..MtmdContextParams::default()
        },
    )?;
    let llamas = format!(
        "data:image/jpeg;base64,{}",
        STANDARD.encode(fs::read(fixtures_dir().join("llamas.jpg"))?)
    );
    let answers = answered_decisions(
        &model,
        backend,
        (*context_params).into_llama_context_params(),
        &DecisionRequest {
            images: vec![llamas.clone(), llamas],
            questions: IndexMap::from([(
                "consistency".to_owned(),
                DecisionQuestion::Choice {
                    criteria: IndexMap::from([
                        ("0".to_owned(), Value::Null),
                        ("1".to_owned(), Value::Null),
                    ]),
                    instructions: Value::Null,
                },
            )]),
            state: json!("Two llamas stand on a grassy hill."),
        },
        Some(&projector),
    )?;

    ensure!(
        matches!(
            answers.get("consistency"),
            Some(DecisionAnswer::Choice { probabilities, .. })
                if probabilities.contains_key("0") && probabilities.contains_key("1")
        ),
        "answers were {answers:?}"
    );

    Ok(())
}
