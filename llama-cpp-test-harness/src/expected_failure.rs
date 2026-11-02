use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use llama_cpp_test_model_provisioning::phase_setup_failure_kind::PhaseSetupFailureKind;

use crate::trial_outcome::TrialOutcome;

/// The failure a trial declares it exists to produce.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedFailure {
    PhaseSetup(PhaseSetupFailureKind),
    Test,
}

impl ExpectedFailure {
    #[must_use]
    pub fn matches(self, outcome: &TrialOutcome<'_>) -> bool {
        match (self, outcome) {
            (Self::PhaseSetup(expected_kind), TrialOutcome::PhaseSetupFailed(setup_failure)) => {
                setup_failure.kind() == expected_kind
            }
            (Self::Test, TrialOutcome::TestFailed(_)) => true,
            (Self::PhaseSetup(_), TrialOutcome::Succeeded | TrialOutcome::TestFailed(_))
            | (Self::Test, TrialOutcome::Succeeded | TrialOutcome::PhaseSetupFailed(_)) => false,
        }
    }
}

impl Display for ExpectedFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhaseSetup(kind) => write!(formatter, "{kind}"),
            Self::Test => formatter.write_str("TestFailure"),
        }
    }
}
