use std::io::Write;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::host_os::HostOs;

/// A build a build script plans from the environment cargo runs it in, then runs.
pub trait PlannedBuild: Sized {
    /// Plans the build of the build script run `environment` describes, on the host the build
    /// script runs on.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`BuildContext::from_environment`] or [`Self::of`].
    fn from_environment(environment: &BuildEnvironment) -> Result<Self, BuildError> {
        Self::of(
            &BuildContext::from_environment(environment, HostOs::current())?,
            environment,
        )
    }

    /// # Errors
    ///
    /// Returns the [`BuildError`] of a step that cannot be planned for `context` and
    /// `environment`.
    fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError>;

    /// Runs every planned step and only then writes the cargo directives to `output`.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of the first step that fails.
    fn run(self, output: &mut impl Write) -> Result<(), BuildError>;
}

#[cfg(test)]
pub mod tests {
    use llama_cpp_error_chain::error_chain::error_chain;

    use super::PlannedBuild;
    use crate::build_environment::BuildEnvironment;

    /// The error chain of running the planned `build`, with what it wrote for cargo.
    ///
    /// # Panics
    ///
    /// Panics when the build writes directives that are not UTF-8.
    #[must_use]
    pub fn run_failure<TBuild: PlannedBuild>(build: TBuild) -> [Option<String>; 2] {
        let mut output = Vec::new();

        [
            build
                .run(&mut output)
                .err()
                .map(|error| error_chain(&error)),
            Some(String::from_utf8(output).unwrap()),
        ]
    }

    impl BuildEnvironment {
        /// The error chain of planning and running `TBuild` for this environment, with what it
        /// wrote for cargo.
        #[must_use]
        pub fn build_failure<TBuild: PlannedBuild>(&self) -> [Option<String>; 2] {
            TBuild::from_environment(self).map_or_else(
                |error| [Some(error_chain(&error)), Some(String::new())],
                run_failure,
            )
        }
    }
}
