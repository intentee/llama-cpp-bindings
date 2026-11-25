use std::fs;
use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use indexmap::IndexMap;
use llama_cpp_bindings_sys::bindings::LLAMA_RS_FAULT_POINT_DECISION_PROMPTS_SHARE_RUNS_OUT_OF_MEMORY;
use llama_cpp_bindings_tests::fixtures_dir::fixtures_dir;
use llama_cpp_bindings_tests::snowy_weather_decision_request::snowy_weather_decision_request;
use llama_cpp_bindings_types::decision_question::DecisionQuestion;
use llama_cpp_bindings_types::decision_request::DecisionRequest;
use llama_cpp_decision::decision_error::DecisionError;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_decision::prepared_decision_request::PreparedDecisionRequest;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;
use serde_json::json;

fn snowy_weather_request_with_a_photo() -> Result<DecisionRequest> {
    let mut request = snowy_weather_decision_request();

    request.images = vec![format!(
        "data:image/jpeg;base64,{}",
        STANDARD.encode(fs::read(fixtures_dir().join("llamas.jpg"))?)
    )];

    Ok(request)
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Kev-0.8B-GGUF", "Kev-0.8B-BF16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 2048,
    n_batch = 2048,
    n_ubatch = 2048,
)]
fn a_choice_without_options_is_refused_with_llama_cpp_reason(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let refusal = PreparedDecisionRequest::new(
        &DecisionModel::new(Arc::clone(model))?,
        &DecisionRequest {
            images: Vec::new(),
            questions: IndexMap::from([(
                "weather".to_owned(),
                DecisionQuestion::Choice {
                    criteria: IndexMap::new(),
                    instructions: json!("What is the weather like?"),
                },
            )]),
            state: json!("Thick snowflakes are falling."),
        },
    );

    ensure!(
        matches!(
            &refusal,
            Err(DecisionError::RequestRejected { message }) if message.contains("must be a non-empty object")
        ),
        "the refusal was {:?}",
        refusal.err()
    );

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
fn a_decision_model_that_cannot_see_refuses_to_lay_out_an_image(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let prepared = PreparedDecisionRequest::new(
        &DecisionModel::new(Arc::clone(model))?,
        &snowy_weather_request_with_a_photo()?,
    )?;

    ensure!(matches!(
        prepared.prompts(None),
        Err(DecisionError::MediaUnsupportedByDecisionModel)
    ));

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
fn a_decision_about_an_image_answered_in_one_prompt_needs_a_projector(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let decision_model = DecisionModel::new(Arc::clone(model))?;

    ensure!(decision_model.is_joint() && decision_model.can_use_images());
    ensure!(matches!(
        PreparedDecisionRequest::new(&decision_model, &snowy_weather_request_with_a_photo()?)?
            .prompts(None),
        Err(DecisionError::MediaRequiresAProjector)
    ));

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
fn prompts_llama_cpp_runs_out_of_memory_sharing_are_not_laid_out(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut request = snowy_weather_decision_request();

    request.questions.insert(
        "freezing".to_owned(),
        DecisionQuestion::Noul {
            criteria: IndexMap::new(),
            instructions: json!("Is it below freezing outside?"),
        },
    );

    let prepared = PreparedDecisionRequest::new(&DecisionModel::new(Arc::clone(model))?, &request)?;
    let _fault = ArmedFault::at(LLAMA_RS_FAULT_POINT_DECISION_PROMPTS_SHARE_RUNS_OUT_OF_MEMORY);

    ensure!(matches!(
        prepared.prompts(None),
        Err(DecisionError::PromptSharingLlamaCppOutOfMemory)
    ));

    Ok(())
}
