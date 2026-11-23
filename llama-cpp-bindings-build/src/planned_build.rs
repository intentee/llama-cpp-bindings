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
