use std::process::ExitCode;

use inventory::submit;
use libtest_mimic::Arguments;

use llama_cpp_bindings::model::llama_load_mode::LlamaLoadMode;
use llama_cpp_test_harness::context_params::ContextParams;
use llama_cpp_test_harness::expected_outcome::ExpectedOutcome;
use llama_cpp_test_harness::harness_run_error::HarnessRunError;
use llama_cpp_test_harness::llama_test_fn::LlamaTestFn;
use llama_cpp_test_harness::llama_test_registration::LlamaTestRegistration;
use llama_cpp_test_harness::no_op::no_op;
use llama_cpp_test_harness::run_to_conclusions::run_to_conclusions;
use llama_cpp_test_model_provisioning::gguf_source::GgufSource;
use llama_cpp_test_model_provisioning::load_key::LoadKey;
use llama_cpp_test_model_provisioning::model_load_params::ModelLoadParams;

const NAME_REGISTERED_TWICE: &str = "a_trial_registered_twice_under_one_name";

const REGISTRATION: LlamaTestRegistration = LlamaTestRegistration {
    name: NAME_REGISTERED_TWICE,
    key: LoadKey {
        model_source: GgufSource::LocalPath("Cargo.toml"),
        mmproj_source: None,
        model_load_params: ModelLoadParams {
            n_gpu_layers: 0,
            load_mode: LlamaLoadMode::Mmap,
        },
    },
    context_params: ContextParams {
        n_ctx: 1,
        n_batch: 1,
        n_ubatch: 1,
        n_seq_max: 1,
        n_threads_batch: None,
        embeddings: false,
    },
    func: LlamaTestFn::Infallible(no_op),
    expected_outcome: ExpectedOutcome::Success,
};

submit! { REGISTRATION }

submit! { REGISTRATION }

fn main() -> ExitCode {
    let refusal = run_to_conclusions(Arguments::from_args()).err();

    if refusal
        == Some(HarnessRunError::DuplicateTrialName {
            name: NAME_REGISTERED_TWICE,
        })
    {
        ExitCode::SUCCESS
    } else {
        eprintln!(
            "duplicate_trial_names: the harness did not refuse the duplicate name: {refusal:?}"
        );

        ExitCode::FAILURE
    }
}
