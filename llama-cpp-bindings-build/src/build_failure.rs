use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;

use llama_cpp_error_chain::error_chain::error_chain;

use crate::build_error::BuildError;

/// A failed native build as a build script's `main` returns it, reported with the whole chain of
/// its causes on one line.
pub struct BuildFailure {
    error: BuildError,
}

impl From<BuildError> for BuildFailure {
    fn from(error: BuildError) -> Self {
        Self { error }
    }
}

impl Debug for BuildFailure {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&error_chain(&self.error))
    }
}

#[cfg(test)]
mod tests {
    use std::env::VarError;

    use llama_cpp_error_chain::error_chain::error_chain;

    use super::BuildFailure;
    use crate::build_error::BuildError;

    #[test]
    fn a_build_failure_reports_every_cause_of_its_error() {
        let failure = BuildFailure::from(BuildError::Environment {
            name: "TARGET",
            source: VarError::NotPresent,
        });

        assert_eq!(
            format!("{failure:?}"),
            error_chain(&BuildError::Environment {
                name: "TARGET",
                source: VarError::NotPresent,
            })
        );
    }
}
