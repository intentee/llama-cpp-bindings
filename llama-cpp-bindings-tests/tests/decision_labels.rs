use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings::error::decision_labels_error::DecisionLabelsError;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_model_without_decision_labels_metadata_has_no_labels(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    ensure!(model.decision_labels()? == Vec::<String>::new());

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("ggml-org/Laya-GGUF", "Laya-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_label_that_is_not_utf8_is_reported_with_its_index(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_MODEL_DECISION_LABEL_NOT_UTF8);

    assert!(matches!(
        model.decision_labels(),
        Err(DecisionLabelsError::LabelNotUtf8 { index: 0, .. })
    ));
}
