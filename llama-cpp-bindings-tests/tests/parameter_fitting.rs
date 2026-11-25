use std::ffi::CString;
use std::num::NonZeroU32;
use std::pin::pin;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;

use llama_cpp_backend::max_devices::max_devices;
use llama_cpp_mtmd::mtmd_context_params::MtmdContextParams;
use llama_cpp_mtmd::mtmd_memory_usage::MtmdMemoryUsage;
use llama_cpp_params::fit_defaults::FitDefaults;
use llama_cpp_params::fit_extra_model::FitExtraModel;
use llama_cpp_params::llama_context_params::LlamaContextParams;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_params::params_error::ParamsError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;
use llama_cpp_test_model_provisioning::gguf_source::GgufSource;

#[llama_test(
    model_source = HuggingFace("unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/GLM-4.7-Flash-GGUF", "GLM-4.7-Flash-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.6-35B-A3B-GGUF", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn fit_params_fits_a_nonzero_context_for_the_model(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    let result = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut margins,
        512,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    let fit = result?;
    ensure!(fit.n_ctx > 0);

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
fn fit_params_aligns_the_extra_model_context_with_the_fitted_one(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    let unfittable_extra_n_ctx = NonZeroU32::MIN;
    let mut extra_params = LlamaModelParams::default();
    let mut extra_context_params =
        LlamaContextParams::default().with_n_ctx(Some(unfittable_extra_n_ctx));
    let mut extra = FitExtraModel {
        model_path,
        model_params: &mut extra_params,
        context_params: &mut extra_context_params,
        shares_model: true,
    };

    let result = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut margins,
        512,
        Some(&mut extra),
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    let fit = result?;
    ensure!(fit.n_ctx > 0);
    ensure!(
        extra_context_params.n_ctx() != Some(unfittable_extra_n_ctx),
        "the llama.cpp fit must overwrite the extra model's context size"
    );
    ensure!(extra_context_params.n_ctx() == context_params.n_ctx());

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
fn fit_params_fits_the_model_beside_the_projector_memory_reserved_on_its_device(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let mmproj_path = GgufSource::HuggingFace {
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        file: "mmproj-F16.gguf",
    }
    .resolve_path()?;
    let mmproj_path_c = CString::new(mmproj_path.as_os_str().as_encoded_bytes())?;

    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default();
    let mut margins = vec![0usize; max_devices()];

    MtmdMemoryUsage::measure(&mmproj_path_c, &MtmdContextParams::default())?
        .add_to_fit_margins(&mut margins);

    ensure!(
        *margins.first().context("margins has no element 0")? != 0,
        "{margins:?}"
    );

    let fit = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut margins,
        512,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    )?;

    ensure!(fit.n_ctx > 0);

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
fn fitting_with_llama_cpp_defaults_keeps_at_least_their_smallest_context(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let fit_defaults = FitDefaults::of_llama_cpp();
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default().with_n_ctx(None);

    let fit = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut fit_defaults.margins(),
        fit_defaults.n_ctx_min,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    )?;

    ensure!(fit.n_ctx >= fit_defaults.n_ctx_min);

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
fn fitting_more_sequences_than_one_batch_outputs_is_refused_before_llama_cpp_measures_them(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let fit_defaults = FitDefaults::of_llama_cpp();
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default()
        .with_n_ctx(None)
        .with_n_batch(1)
        .with_n_seq_max(2);

    let refusal = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut fit_defaults.margins(),
        fit_defaults.n_ctx_min,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    ensure!(
        refusal
            == Err(ParamsError::FitSequencesExceedOutputCapacity {
                n_seq_max: 2,
                output_capacity: 1,
            }),
        "{refusal:?}"
    );

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
fn fitting_an_extra_model_with_more_sequences_than_one_batch_outputs_is_refused_before_llama_cpp_measures_it(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let fit_defaults = FitDefaults::of_llama_cpp();
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default().with_n_ctx(None);
    let mut extra_params = LlamaModelParams::default();
    let mut extra_context_params = LlamaContextParams::default()
        .with_n_batch(1)
        .with_n_seq_max(2);

    let refusal = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut fit_defaults.margins(),
        fit_defaults.n_ctx_min,
        Some(&mut FitExtraModel {
            model_path,
            model_params: &mut extra_params,
            context_params: &mut extra_context_params,
            shares_model: true,
        }),
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    ensure!(
        refusal
            == Err(ParamsError::FitExtraModelSequencesExceedOutputCapacity {
                n_seq_max: 2,
                output_capacity: 1,
            }),
        "{refusal:?}"
    );

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
fn fitting_a_context_with_an_attention_type_llama_cpp_does_not_define_is_refused(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) -> Result<()> {
    let fit_defaults = FitDefaults::of_llama_cpp();
    let mut params = pin!(LlamaModelParams::default());
    let mut context_params = LlamaContextParams::default().with_n_ctx(None);
    unsafe { (*context_params.as_mut_ffi()).attention_type = 99 };

    let refusal = params.as_mut().fit_params(
        model_path,
        &mut context_params,
        &mut fit_defaults.margins(),
        fit_defaults.n_ctx_min,
        None,
        llama_cpp_bindings_sys::bindings::GGML_LOG_LEVEL_NONE,
    );

    ensure!(
        refusal == Err(ParamsError::UnknownAttentionType(99)),
        "{refusal:?}"
    );

    Ok(())
}
