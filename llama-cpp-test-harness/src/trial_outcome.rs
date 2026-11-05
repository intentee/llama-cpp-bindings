use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use anyhow::Error;

use llama_cpp_test_model_provisioning::phase_setup_error::PhaseSetupError;

/// How a trial actually ended.
#[derive(Debug)]
pub enum TrialOutcome<'setup> {
    Succeeded,
    PhaseSetupFailed(&'setup PhaseSetupError),
    TestFailed(Error),
}

impl Display for TrialOutcome<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Succeeded => formatter.write_str("the trial succeeded"),
            Self::PhaseSetupFailed(setup_failure) => {
                write!(formatter, "phase setup failed: {setup_failure}")
            }
            Self::TestFailed(test_failure) => write!(formatter, "{test_failure:#}"),
        }
    }
}
