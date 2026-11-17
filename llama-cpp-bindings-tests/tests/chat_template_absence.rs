use llama_cpp_model::model_error::ModelError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn a_model_without_a_chat_template_has_no_chat_templates(
    &LlamaFixture { model, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(
        model.chat_templates().err(),
        Some(ModelError::ModelHasNoChatTemplate)
    );
}
