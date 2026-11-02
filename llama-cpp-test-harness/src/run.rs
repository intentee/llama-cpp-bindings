use std::process::ExitCode;

use libtest_mimic::Arguments;
use libtest_mimic::Conclusion;

use crate::run_to_conclusions::run_to_conclusions;

fn aggregate_exit_code(conclusions: &[Conclusion]) -> ExitCode {
    if conclusions.iter().any(Conclusion::has_failed) {
        ExitCode::from(101)
    } else {
        ExitCode::SUCCESS
    }
}

#[must_use]
pub fn run() -> ExitCode {
    match run_to_conclusions(Arguments::from_args()) {
        Ok(conclusions) => aggregate_exit_code(&conclusions),
        Err(error) => {
            eprintln!("llama-cpp-test-harness: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::process::ExitCode;
    use std::sync::PoisonError;

    use libtest_mimic::Conclusion;

    use llama_cpp_bindings::llama_backend::LlamaBackend;
    use llama_cpp_test_model_provisioning::backend_init_gate::BACKEND_INIT_GATE;

    use super::aggregate_exit_code;
    use super::run;

    fn passing_conclusion() -> Conclusion {
        Conclusion {
            num_filtered_out: 0,
            num_passed: 1,
            num_failed: 0,
            num_ignored: 0,
            num_measured: 0,
        }
    }

    fn failing_conclusion() -> Conclusion {
        Conclusion {
            num_filtered_out: 0,
            num_passed: 0,
            num_failed: 1,
            num_ignored: 0,
            num_measured: 0,
        }
    }

    #[test]
    fn aggregate_exit_code_zero_when_all_pass() {
        let code = aggregate_exit_code(&[passing_conclusion(), passing_conclusion()]);

        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn aggregate_exit_code_non_zero_when_any_fails() {
        let code = aggregate_exit_code(&[passing_conclusion(), failing_conclusion()]);

        assert_eq!(code, ExitCode::from(101));
    }

    #[test]
    fn aggregate_exit_code_reports_success_when_no_phase_ran() {
        let code = aggregate_exit_code(&[]);

        assert_eq!(code, ExitCode::from(0));
    }

    #[test]
    fn run_returns_exit_code_two_when_backend_init_fails() {
        let _gate = BACKEND_INIT_GATE
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let _hold = LlamaBackend::init().expect("first init must succeed");

        let code = run();

        assert_eq!(code, ExitCode::from(2));
    }
}
