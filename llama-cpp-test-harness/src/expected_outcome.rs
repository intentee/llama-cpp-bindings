use libtest_mimic::Failed;
use libtest_mimic::Trial;

use crate::expected_failure::ExpectedFailure;
use crate::trial_outcome::TrialOutcome;

/// How a trial is declared to end: by succeeding, or by producing one exact failure on purpose,
/// for the stated reason.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedOutcome {
    Success,
    Failure {
        failure: ExpectedFailure,
        reason: &'static str,
    },
}

impl ExpectedOutcome {
    /// Marks a trial that is declared to fail with the failure it expects and why.
    #[must_use]
    pub fn label(self, trial: Trial) -> Trial {
        match self {
            Self::Success => trial,
            Self::Failure { failure, reason } => {
                trial.with_kind(format!("expected {failure}: {reason}"))
            }
        }
    }

    /// # Errors
    ///
    /// Returns [`Failed`] when `outcome` is not the declared one: a failure of a trial declared to
    /// succeed, or a success or a different failure of a trial declared to fail.
    pub fn judge(self, outcome: TrialOutcome<'_>) -> Result<(), Failed> {
        match self {
            Self::Success => match outcome {
                TrialOutcome::Succeeded => Ok(()),
                failure @ (TrialOutcome::PhaseSetupFailed(_) | TrialOutcome::TestFailed(_)) => {
                    Err(Failed::from(failure))
                }
            },
            Self::Failure { failure, .. } if failure.matches(&outcome) => Ok(()),
            Self::Failure { failure, .. } => {
                Err(Failed::from(format!("expected {failure}, but {outcome}")))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use anyhow::anyhow;

    use llama_cpp_test_model_provisioning::phase_setup_error::PhaseSetupError;
    use llama_cpp_test_model_provisioning::phase_setup_failure_kind::PhaseSetupFailureKind;

    use super::ExpectedOutcome;
    use crate::expected_failure::ExpectedFailure;
    use crate::trial_outcome::TrialOutcome;

    const EXPECTED_SETUP_FAILURE: ExpectedOutcome = ExpectedOutcome::Failure {
        failure: ExpectedFailure::PhaseSetup(PhaseSetupFailureKind::MmprojPathNotUnicode),
        reason: "the mmproj path is not Unicode on purpose",
    };

    const EXPECTED_TEST_FAILURE: ExpectedOutcome = ExpectedOutcome::Failure {
        failure: ExpectedFailure::Test,
        reason: "the body bails on purpose",
    };

    fn setup_failure() -> PhaseSetupError {
        PhaseSetupError::MmprojPathNotUnicode {
            mmproj_path: PathBuf::from("/models/mmproj.gguf"),
        }
    }

    fn verdict(expected_outcome: ExpectedOutcome, outcome: TrialOutcome<'_>) -> Result<(), String> {
        expected_outcome
            .judge(outcome)
            .map_err(|failed| failed.message().unwrap_or_default().to_owned())
    }

    #[test]
    fn a_success_meets_an_expected_success() {
        assert_eq!(
            verdict(ExpectedOutcome::Success, TrialOutcome::Succeeded),
            Ok(())
        );
    }

    #[test]
    fn a_failure_misses_an_expected_success_with_what_failed() {
        let setup_failure = setup_failure();

        assert_eq!(
            [
                verdict(
                    ExpectedOutcome::Success,
                    TrialOutcome::PhaseSetupFailed(&setup_failure)
                ),
                verdict(
                    ExpectedOutcome::Success,
                    TrialOutcome::TestFailed(anyhow!("the body failed"))
                ),
            ],
            [
                Err(
                    "phase setup failed: the mmproj path /models/mmproj.gguf is not Unicode"
                        .to_owned()
                ),
                Err("the body failed".to_owned()),
            ]
        );
    }

    #[test]
    fn the_declared_failure_meets_its_expectation() {
        let setup_failure = setup_failure();

        assert_eq!(
            [
                verdict(
                    EXPECTED_SETUP_FAILURE,
                    TrialOutcome::PhaseSetupFailed(&setup_failure)
                ),
                verdict(
                    EXPECTED_TEST_FAILURE,
                    TrialOutcome::TestFailed(anyhow!("the body bailed"))
                ),
            ],
            [Ok(()), Ok(())]
        );
    }

    #[test]
    fn a_success_misses_an_expected_failure() {
        assert_eq!(
            verdict(EXPECTED_SETUP_FAILURE, TrialOutcome::Succeeded),
            Err("expected MmprojPathNotUnicode, but the trial succeeded".to_owned())
        );
    }

    #[test]
    fn a_different_failure_misses_an_expected_failure() {
        let setup_failure = setup_failure();

        assert_eq!(
            [
                verdict(
                    ExpectedOutcome::Failure {
                        failure: ExpectedFailure::PhaseSetup(PhaseSetupFailureKind::ModelDownload),
                        reason: "the repository does not exist on purpose",
                    },
                    TrialOutcome::PhaseSetupFailed(&setup_failure)
                ),
                verdict(
                    EXPECTED_SETUP_FAILURE,
                    TrialOutcome::TestFailed(anyhow!("the body failed"))
                ),
                verdict(
                    EXPECTED_TEST_FAILURE,
                    TrialOutcome::PhaseSetupFailed(&setup_failure)
                ),
                verdict(EXPECTED_TEST_FAILURE, TrialOutcome::Succeeded),
            ],
            [
                Err(
                    "expected ModelDownload, but phase setup failed: the mmproj path \
                     /models/mmproj.gguf is not Unicode"
                        .to_owned()
                ),
                Err("expected MmprojPathNotUnicode, but the body failed".to_owned()),
                Err(
                    "expected TestFailure, but phase setup failed: the mmproj path \
                     /models/mmproj.gguf is not Unicode"
                        .to_owned()
                ),
                Err("expected TestFailure, but the trial succeeded".to_owned()),
            ]
        );
    }
}
