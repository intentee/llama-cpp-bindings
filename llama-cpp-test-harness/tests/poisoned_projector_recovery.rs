use std::process::ExitCode;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;

use llama_cpp_test_harness::llama_fixture::LlamaFixture;
use llama_cpp_test_harness::run_to_conclusions::run_to_conclusions;
use llama_cpp_test_harness_macros::llama_test;

#[llama_test(
    model_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf"),
    n_gpu_layers = -1,
    load_mode = Mmap,
    n_ctx = 512,
    n_batch = 128,
    n_ubatch = 64,
    mmproj_source = HuggingFace("unsloth/Qwen3.5-0.8B-GGUF", "mmproj-F16.gguf")
)]
fn a_trial_fails_an_assertion_while_holding_the_projector(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) {
    assert!(
        mtmd_context.is_none(),
        "this trial fails on purpose while it holds the projector, so its panic poisons the \
         projector lock"
    );
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
fn the_next_trial_of_that_projector_receives_it(
    &LlamaFixture { mtmd_context, .. }: &LlamaFixture<'_>,
) {
    assert!(
        mtmd_context.is_some(),
        "the trial after one that panicked holding the projector must still receive it"
    );
}

const ONLY_THE_PANICKING_TRIAL_FAILS: Conclusion = Conclusion {
    num_filtered_out: 0,
    num_passed: 1,
    num_failed: 1,
    num_ignored: 0,
    num_measured: 0,
};

fn main() -> ExitCode {
    let conclusions = match run_to_conclusions(Arguments::from_args()) {
        Ok(conclusions) => conclusions,
        Err(error) => {
            eprintln!("poisoned_projector_recovery: unexpected harness setup failure: {error}");

            return ExitCode::FAILURE;
        }
    };

    if conclusions == [ONLY_THE_PANICKING_TRIAL_FAILS] {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "poisoned_projector_recovery: want the trial after the one that panicked holding the \
             projector to pass; got {conclusions:?}"
        );

        ExitCode::FAILURE
    }
}
