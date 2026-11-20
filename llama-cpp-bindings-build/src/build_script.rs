use std::io;
use std::io::Write;
use std::process::ExitCode;

use llama_cpp_error_chain::error_chain::error_chain;

use crate::build_environment::BuildEnvironment;
use crate::native_build::NativeBuild;

/// The build script of the bindings, with the stream cargo reads its directives from and the
/// stream that reports why a build failed.
pub struct BuildScript<TDirectives, TDiagnostics> {
    pub directives: TDirectives,
    pub diagnostics: TDiagnostics,
}

impl<TDirectives: Write, TDiagnostics: Write> BuildScript<TDirectives, TDiagnostics> {
    /// Builds llama.cpp and its bindings for the build script run `environment` describes,
    /// writing the cargo directives to `directives` and the error chain of a failed build to
    /// `diagnostics`.
    ///
    /// # Errors
    ///
    /// Returns the [`io::Error`] of writing the error chain of a failed build to `diagnostics`.
    pub fn run(&mut self, environment: &BuildEnvironment) -> Result<ExitCode, io::Error> {
        match NativeBuild::from_environment(environment)
            .and_then(|native_build| native_build.run(&mut self.directives))
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

    #[test]
    fn a_failed_build_reports_its_error_chain_emits_no_directive_and_fails() {
        let mut without_target = cargo_environment(BitFlags::empty(), &[]);
        let mut build_script = BuildScript {
            directives: Vec::new(),
            diagnostics: Vec::new(),
        };

        without_target.variables.remove(OsStr::new("TARGET"));

        assert_eq!(
            build_script.run(&without_target).ok(),
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
