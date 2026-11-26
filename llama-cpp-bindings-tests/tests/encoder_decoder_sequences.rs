use std::pin::pin;

use anyhow::Result;
use anyhow::ensure;

use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_params::fit_defaults::FitDefaults;
use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_model_params::LlamaModelParams;
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
fn an_encoder_decoder_model_fits_a_context_for_several_sequences(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let fit_defaults = FitDefaults::of_llama_cpp();
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(None)
        .with_n_seq_max(2);

    let fitted = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut fit_defaults.margins(),
        fit_defaults.n_ctx_min,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    ensure!(fitted.is_ok(), "{fitted:?}");

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("Xiaojian9992024/t5-small-GGUF", "t5-small.bf16.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 512,
)]
fn an_encoder_decoder_model_constructs_a_context_for_several_sequences(
    &LlamaFixture {
        model,
        backend,
        context_params,
        ..
    }: &LlamaFixture<'_>,
) -> Result<()> {
    let context = LlamaContext::from_model(
        model,
        backend,
        (*context_params)
            .into_llama_context_params()
            .with_n_seq_max(2),
    );

    ensure!(context.is_ok(), "{:?}", context.err());

    Ok(())
}
