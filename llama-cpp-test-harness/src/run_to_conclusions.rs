use std::sync::Arc;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;

use llama_cpp_bindings::llama_backend::LlamaBackend;
use llama_cpp_bindings::log_options::LogOptions;
use llama_cpp_bindings::send_logs_to_log::send_logs_to_log;

use crate::execution_plan::ExecutionPlan;
use crate::harness_run_error::HarnessRunError;
use crate::parse_harness_arguments::parse_harness_arguments;

/// # Errors
///
/// Returns [`HarnessRunError`] when the CLI arguments conflict with the harness's single-thread
/// requirement or the llama backend cannot be initialised. Surfacing these as a typed error keeps
/// the failure explicit instead of aborting the process with a panic.
pub fn run_to_conclusions(arguments: Arguments) -> Result<Vec<Conclusion>, HarnessRunError> {
    let arguments = parse_harness_arguments(arguments)?;
    let backend = Arc::new(LlamaBackend::init()?);
    let plan = ExecutionPlan::from_inventory();
    if plan.requests_void_logs() {
        send_logs_to_log(LogOptions::default().with_logs_enabled(false));
    }

    Ok(plan.run(&backend, &arguments))
}

#[cfg(test)]
mod tests {
    use std::sync::PoisonError;

    use libtest_mimic::Arguments;

    use llama_cpp_bindings::error::llama_cpp_error::LlamaCppError;
    use llama_cpp_bindings::llama_backend::LlamaBackend;

    use super::run_to_conclusions;
    use crate::harness_arguments_error::HarnessArgumentsError;
    use crate::harness_run_error::HarnessRunError;
    use crate::test_backend_gate::BACKEND_INIT_GATE;

    #[test]
    fn conflicting_arguments_stop_the_run_before_the_backend_starts() {
        assert_eq!(
            run_to_conclusions(Arguments {
                test_threads: Some(8),
                ..Arguments::default()
            })
            .err(),
            Some(HarnessRunError::ArgumentParsing(
                HarnessArgumentsError::ConflictingTestThreads { requested: 8 }
            ))
        );
    }

    #[test]
    fn an_already_initialized_backend_stops_the_run_with_a_backend_init_error() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let _hold = LlamaBackend::init().expect("first init must succeed");

        assert_eq!(
            run_to_conclusions(Arguments::default()).err(),
            Some(HarnessRunError::BackendInit(
                LlamaCppError::BackendAlreadyInitialized
            ))
        );
    }

    #[test]
    fn empty_inventory_yields_no_conclusions() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        assert_eq!(run_to_conclusions(Arguments::default()), Ok(Vec::new()));
    }
}
