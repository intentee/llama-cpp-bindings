use std::sync::Arc;

use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::model_declaring_decision_type::model_declaring_decision_type;
use llama_cpp_decision::decision_error::DecisionError;
use llama_cpp_decision::decision_model::DecisionModel;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_without_a_decision_type_serves_no_decisions(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(matches!(
        DecisionModel::new(Arc::clone(model)),
        Err(DecisionError::ModelIsNotADecisionModel)
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_model_declaring_a_decision_type_llama_cpp_does_not_know_serves_no_decisions(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_declaring_decision_type(
        fixture,
        c"not-a-decision-type",
        "llama_test_unknown_decision_type.gguf",
    )?;

    ensure!(matches!(
        DecisionModel::new(model),
        Err(DecisionError::DecisionTypeUnsupportedByLlamaCpp)
    ));

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_decision_model_without_its_decision_template_is_rejected(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = model_declaring_decision_type(
        fixture,
        c"kev",
        "llama_test_decision_model_without_template.gguf",
    )?;

    ensure!(matches!(
        DecisionModel::new(model),
        Err(DecisionError::ModelRejected { .. })
    ));

    Ok(())
}
