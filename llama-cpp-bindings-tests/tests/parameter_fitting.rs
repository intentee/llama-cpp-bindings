use std::ffi::CString;
use std::num::NonZeroU32;
use std::pin::pin;

use anyhow::Result;
use llama_cpp_bindings::context::params::LlamaContextParams;
use llama_cpp_bindings::max_devices::max_devices;
use llama_cpp_bindings::model::params::LlamaModelParams;
use llama_cpp_bindings::model::params::fit_extra_model::FitExtraModel;
use llama_cpp_bindings::mtmd::mtmd_context_params::MtmdContextParams;
use llama_cpp_bindings::mtmd::mtmd_memory_usage::MtmdMemoryUsage;
use llama_cpp_test_harness::GgufSource;
use llama_cpp_test_harness::LlamaFixture;
use llama_cpp_test_harness::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn fit_params_succeeds_with_test_model(fixture: &LlamaFixture<'_>) -> Result<()> {
    let model_path_c = CString::new(fixture.model_path.as_os_str().as_encoded_bytes())?;

    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    let result = params.as_mut().fit_params(
        &model_path_c,
        &mut context_params,
        &mut margins,
        512,
        None,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
    );

    let fit = result?;
    assert!(fit.n_ctx > 0);

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn fit_params_aligns_the_extra_model_context_with_the_fitted_one(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model_path_c = CString::new(fixture.model_path.as_os_str().as_encoded_bytes())?;

    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    let unfittable_extra_n_ctx = NonZeroU32::MIN;
    let mut extra_params = pin!(LlamaModelParams::default());
    let mut extra_context_params =
        LlamaContextParams::default().with_n_ctx(Some(unfittable_extra_n_ctx));
    let mut extra = FitExtraModel {
        model_path: &model_path_c,
        model_params: extra_params.as_mut(),
        context_params: &mut extra_context_params,
        shares_model: true,
    };

    let result = params.as_mut().fit_params(
        &model_path_c,
        &mut context_params,
        &mut margins,
        512,
        Some(&mut extra),
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
    );

    let fit = result?;
    assert!(fit.n_ctx > 0);
    assert_ne!(
        extra_context_params.n_ctx(),
        Some(unfittable_extra_n_ctx),
        "the llama.cpp fit must overwrite the extra model's context size"
    );
    assert_eq!(
        extra_context_params.context_params.n_ctx,
        context_params.context_params.n_ctx
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = 999,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn fit_params_fits_the_model_beside_the_projector_memory_reserved_on_its_device(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let mmproj_path = GgufSource::HuggingFace {
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        file: "mmproj-F16.gguf",
    }
    .resolve_path()?;
    let model_path_c = CString::new(fixture.model_path.as_os_str().as_encoded_bytes())?;
    let mmproj_path_c = CString::new(mmproj_path.as_os_str().as_encoded_bytes())?;

    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    MtmdMemoryUsage::measure(&mmproj_path_c, &MtmdContextParams::default())?
        .add_to_fit_margins(&mut margins);

    assert_ne!(margins[0], 0, "{margins:?}");

    let fit = params.as_mut().fit_params(
        &model_path_c,
        &mut context_params,
        &mut margins,
        512,
        None,
        llama_cpp_bindings_sys::GGML_LOG_LEVEL_NONE,
    )?;

    assert!(fit.n_ctx > 0);

    Ok(())
}
