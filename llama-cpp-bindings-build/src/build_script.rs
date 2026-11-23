use std::io;
use std::io::Stderr;
use std::io::Stdout;
use std::io::Write;
use std::process::ExitCode;

use llama_cpp_error_chain::error_chain::error_chain;

use crate::build_environment::BuildEnvironment;
use crate::planned_build::PlannedBuild;

/// A build script, with the stream cargo reads its directives from and the stream that reports
/// why a build failed.
pub struct BuildScript<TDirectives, TDiagnostics> {
    pub directives: TDirectives,
    pub diagnostics: TDiagnostics,
}

impl BuildScript<Stdout, Stderr> {
    /// Plans and runs `TBuild` for the build script of the running process, writing the cargo
    /// directives to standard output and the error chain of a failed build to standard error.
    ///
    /// # Errors
    ///
    /// Returns the [`io::Error`] of writing the error chain of a failed build to standard error.
    pub fn run_in_process<TBuild: PlannedBuild>() -> Result<ExitCode, io::Error> {
        Self {
            directives: io::stdout(),
            diagnostics: io::stderr(),
        }
        .run::<TBuild>(&BuildEnvironment::of_process())
    }
}

impl<TDirectives: Write, TDiagnostics: Write> BuildScript<TDirectives, TDiagnostics> {
    /// Plans and runs `TBuild` for the build script run `environment` describes, writing the
    /// cargo directives to `directives` and the error chain of a failed build to `diagnostics`.
    ///
    /// # Errors
    ///
    /// Returns the [`io::Error`] of writing the error chain of a failed build to `diagnostics`.
    pub fn run<TBuild: PlannedBuild>(
        &mut self,
        environment: &BuildEnvironment,
    ) -> Result<ExitCode, io::Error> {
        match TBuild::from_environment(environment)
            .and_then(|build| build.run(&mut self.directives))
        {
            Ok(()) => Ok(ExitCode::SUCCESS),
            Err(build_error) => writeln!(self.diagnostics, "Error: {}", error_chain(&build_error))
                .map(|()| ExitCode::FAILURE),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::process::ExitCode;

    use enumflags2::BitFlags;

    use super::BuildScript;
    use crate::build_context::tests::cargo_environment;
    use crate::native_library_build::NativeLibraryBuild;

    #[test]
    fn a_failed_build_reports_its_error_chain_emits_no_directive_and_fails() {
        let mut without_target = cargo_environment(BitFlags::empty(), &[]);
        let mut build_script = BuildScript {
            directives: Vec::new(),
            diagnostics: Vec::new(),
        };

        without_target.variables.remove(OsStr::new("TARGET"));

        assert_eq!(
            build_script.run::<NativeLibraryBuild>(&without_target).ok(),
            Some(ExitCode::FAILURE)
        );
        assert_eq!(
            [build_script.directives, build_script.diagnostics],
            [
                Vec::new(),
                b"Error: environment variable TARGET could not be read: environment variable not \
                  found\n"
                    .to_vec()
            ]
        );
    }
}
