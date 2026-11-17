use anyhow::Result;

use llama_cpp_bindings::model::decision_type::DecisionType;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("ggml-org/OpenJev-GGUF", "OpenJev-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn openjev_is_detected_as_an_openjev_decision_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(fixture.model.decision_type()?, DecisionType::OpenJev);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn laya_is_detected_as_a_laya_decision_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(fixture.model.decision_type()?, DecisionType::Laya);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Clef-Flash-GGUF", "Clef-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn clef_flash_is_detected_as_a_clef_decision_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(fixture.model.decision_type()?, DecisionType::Clef);

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
fn a_text_generation_model_is_not_a_decision_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    assert_eq!(fixture.model.decision_type()?, DecisionType::None);

    Ok(())
}
