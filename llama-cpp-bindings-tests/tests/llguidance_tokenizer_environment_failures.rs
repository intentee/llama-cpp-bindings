use anyhow::Result;
use anyhow::ensure;

use llama_cpp_bindings_tests::yes_or_no_regex::YES_OR_NO_REGEX;
use llama_cpp_fault_injection::armed_fault::ArmedFault;
use llama_cpp_model::llama_model::LlamaModel;
use llama_cpp_model::model_error::ModelError;
use llama_cpp_params::llama_model_params::LlamaModelParams;
use llama_cpp_sampler::llama_sampler::LlamaSampler;
use llama_cpp_sampler::sampler_error::SamplerError;
use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness_macros::llama_test;

fn freshly_loaded(
    &LlamaFixture {
        backend,
        model_path,
        ..
    }: &LlamaFixture<'_>,
) -> Result<LlamaModel> {
    Ok(LlamaModel::load_from_file(
        backend,
        model_path,
        &LlamaModelParams::default(),
    )?)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn creating_a_sampler_reports_a_vocabulary_llama_cpp_cannot_render(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = freshly_loaded(fixture)?;
    let _fault = ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_TOKEN_TO_PIECE_FAILS);

    ensure!(
        LlamaSampler::llguidance(&model, "regex", YES_OR_NO_REGEX).err()
            == Some(SamplerError::TokEnvUnavailable(
                ModelError::PieceInsufficientBufferSpace(-34)
            ))
    );

    Ok(())
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 512,
    n_ubatch = 128,
)]
fn the_tokenizer_environment_reports_a_control_token_llama_cpp_cannot_render(
    fixture: &LlamaFixture<'_>,
) -> Result<()> {
    let model = freshly_loaded(fixture)?;
    let _fault =
        ArmedFault::at(llama_cpp_bindings_sys::LLAMA_RS_FAULT_POINT_SPECIAL_TOKEN_TO_PIECE_FAILS);

    ensure!(
        model.approximate_tok_env().err() == Some(ModelError::PieceInsufficientBufferSpace(-34))
    );

    Ok(())
}
