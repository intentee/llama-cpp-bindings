use std::process::ExitCode;

use anyhow::Result;
use anyhow::bail;
use libtest_mimic::Arguments;

use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness::no_op::no_op;
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
const fn phase_a_passing_trial_that_only_needs_its_model_to_load(fixture: &LlamaFixture<'_>) {
    no_op(fixture);
}

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_a_second_passing_trial(fixture: &LlamaFixture<'_>) {
    assert_eq!(fixture.context_params.n_ctx, 512);
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
fn phase_a_intentionally_failing_trial(_fixture: &LlamaFixture<'_>) -> Result<()> {
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
fn phase_b_first_passing_trial(fixture: &LlamaFixture<'_>) {
    assert!(fixture.mtmd_context.is_none());
}

#[llama_test(
    model_source = HuggingFace("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64
)]
fn phase_b_second_passing_trial(fixture: &LlamaFixture<'_>) {
    assert_eq!(fixture.context_params.n_ubatch, 64);
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
    mmproj_source = LocalPath("/nonexistent/llama-cpp-test-harness/no-such-mmproj.gguf"),
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
fn phase_a_trial_also_scheduled_into_every_setup_failure_phase(fixture: &LlamaFixture<'_>) {
    assert!(fixture.model_path.exists());
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
fn phase_d_passing_trial_with_a_loaded_mmproj(fixture: &LlamaFixture<'_>) {
    assert!(
        fixture.mtmd_context.is_some(),
        "mmproj_file declared, but fixture.mtmd_context is None",
    );
}

const EXPECTED_PHASES: usize = 7;

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
