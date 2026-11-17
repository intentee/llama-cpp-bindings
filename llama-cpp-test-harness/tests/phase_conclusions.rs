use std::process::ExitCode;

use anyhow::Result;
use anyhow::bail;
use libtest_mimic::Arguments;

use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness::llama_test_fn::LlamaTestFn;
use llama_cpp_test_harness::run_to_conclusions::run_to_conclusions;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
)]
fn a_trial_passes_once_its_model_has_loaded(fixture: &LlamaFixture<'_>) -> Result<()> {
    LlamaTestFn::no_op().run(fixture)
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn a_trial_receives_its_declared_context_size(
    &LlamaFixture { context_params, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(context_params.n_ctx, 512);
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    expected_failure = TestFailure("the body bails to exercise reporting a failing test"),
)]
fn a_trial_expected_to_fail_passes_when_its_body_fails(_fixture: &LlamaFixture<'_>) -> Result<()> {
    bail!("intentional failure to exercise the trial-failure dispatch path");
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn an_embedding_model_trial_without_an_mmproj_sees_no_projector(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) {
    assert!(mtmd_context.is_none());
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn a_trial_receives_its_declared_micro_batch_size(
    &LlamaFixture { context_params, .. }: &LlamaFixture<'_>,
) {
    assert_eq!(context_params.n_ubatch, 64);
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
#[llama_test(
    model_source = HuggingFace("intentee-test-harness/does-not-exist", "no-such-file.gguf"),
    n_gpu_layers = 0,
    load_mode = Mmap,
    n_ctx = 1,
    n_batch = 1,
    n_ubatch = 1,
    expected_failure = SetupFailure(
        ModelDownload,
        "the model repository does not exist, so the phase cannot download its model"
    ),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("intentee-test-harness/does-not-exist", "no-such-mmproj.gguf"),
    expected_failure = SetupFailure(
        MmprojDownload,
        "the mmproj repository does not exist, so the phase cannot download its mmproj"
    ),
)]
#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = LocalPath("/nonexistent/llama-cpp-test-harness/no-such-local-mmproj.gguf"),
    expected_failure = SetupFailure(
        MmprojLoad,
        "the local mmproj file does not exist, so llama.cpp cannot load it"
    ),
)]
#[llama_test(
    model_source = LocalPath("Cargo.toml"),
    n_gpu_layers = 0,
    load_mode = Mmap,
    n_ctx = 1,
    n_batch = 1,
    n_ubatch = 1,
    expected_failure = SetupFailure(
        ModelLoad,
        "Cargo.toml is not a GGUF model, so llama.cpp cannot load it"
    ),
)]
fn a_trial_sees_its_model_file_unless_its_setup_is_expected_to_fail(
    &LlamaFixture { model_path, .. }: &LlamaFixture<'_>,
) {
    assert!(model_path.exists());
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf")
)]
fn a_trial_declaring_an_mmproj_sees_its_projector(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) {
    assert!(
        mtmd_context.is_some(),
        "mmproj_file declared, but fixture.mtmd_context is None",
    );
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn trial_without_an_mmproj_sees_no_projector_after_its_model_loaded_one(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) {
    assert!(
        mtmd_context.is_none(),
        "no mmproj_source declared, but fixture.mtmd_context is Some",
    );
}

const EXPECTED_PHASES: usize = 4;

fn main() -> ExitCode {
    let conclusions = match run_to_conclusions(Arguments::from_args()) {
        Ok(conclusions) => conclusions,
        Err(error) => {
            eprintln!("phase_conclusions: unexpected harness setup failure: {error}");

            return ExitCode::FAILURE;
        }
    };
    let phases = conclusions.len();
    let phases_with_a_failed_trial = conclusions
        .iter()
        .filter(|conclusion| conclusion.has_failed())
        .count();

    if phases == EXPECTED_PHASES && phases_with_a_failed_trial == 0 {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "phase_conclusions: want {EXPECTED_PHASES} phases without a failed trial; got \
             {phases} phases, {phases_with_a_failed_trial} of them with a failed trial"
        );

        ExitCode::FAILURE
    }
}
