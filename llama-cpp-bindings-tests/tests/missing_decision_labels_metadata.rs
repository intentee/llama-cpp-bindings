use anyhow::Result;
use anyhow::ensure;

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
