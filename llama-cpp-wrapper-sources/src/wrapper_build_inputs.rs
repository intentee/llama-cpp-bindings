use std::path::PathBuf;

use crate::compile_commands_file::CompileCommandsFile;
use crate::wrapper_sources_error::WrapperSourcesError;
use crate::wrapper_sources_response_file::WrapperSourcesResponseFile;

fn required_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &'static str,
) -> Result<String, WrapperSourcesError> {
    arguments
        .next()
        .ok_or(WrapperSourcesError::MissingArgument { name })
}

/// The compilation database and the response file the C++ linters read the wrapper sources from.
pub struct WrapperBuildInputs {
    compile_commands_file: CompileCommandsFile,
    response_file: WrapperSourcesResponseFile,
}

impl WrapperBuildInputs {
    /// Reads `<sys-dir> <compile-commands-path> <response-file-path>` from the command line
    /// arguments that follow the program name.
    ///
    /// # Errors
    ///
    /// Returns [`WrapperSourcesError::MissingArgument`] naming the first argument that is absent.
    pub fn from_arguments(
        mut arguments: impl Iterator<Item = String>,
    ) -> Result<Self, WrapperSourcesError> {
        let sys_dir = PathBuf::from(required_argument(&mut arguments, "sys-dir")?);
        let compile_commands_path = required_argument(&mut arguments, "compile-commands-path")?;
        let response_file_path = required_argument(&mut arguments, "response-file-path")?;

        Ok(Self {
            compile_commands_file: CompileCommandsFile {
                output_path: PathBuf::from(compile_commands_path),
                sys_dir: sys_dir.clone(),
            },
            response_file: WrapperSourcesResponseFile {
                output_path: PathBuf::from(response_file_path),
                sys_dir,
            },
        })
    }

    /// # Errors
    ///
    /// Returns [`WrapperSourcesError`] when the compilation database or the response file cannot
    /// be written.
    pub fn write(&self) -> Result<(), WrapperSourcesError> {
        self.compile_commands_file.write()?;
        self.response_file.write()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::WrapperBuildInputs;

    const ARGUMENTS: [&str; 3] = [
        "/repo/llama-cpp-bindings-sys",
        "/repo/target/compile_commands.json",
        "/repo/target/wrapper_sources.rsp",
    ];

    fn missing_argument_message(given: &[&str]) -> Option<String> {
        WrapperBuildInputs::from_arguments(given.iter().map(|argument| (*argument).to_owned()))
            .err()
            .map(|error| error.to_string())
    }

    #[test]
    fn the_first_absent_argument_is_named() {
        assert_eq!(
            [
                missing_argument_message(&ARGUMENTS[..0]),
                missing_argument_message(&ARGUMENTS[..1]),
                missing_argument_message(&ARGUMENTS[..2]),
                missing_argument_message(&ARGUMENTS),
            ],
            [
                Some("required command line argument <sys-dir> is missing".to_owned()),
                Some(
                    "required command line argument <compile-commands-path> is missing".to_owned()
                ),
                Some("required command line argument <response-file-path> is missing".to_owned()),
                None,
            ]
        );
    }

    #[test]
    fn both_files_are_written_for_the_sys_dir_at_the_paths_given_in_order() {
        let inputs = WrapperBuildInputs::from_arguments(
            ARGUMENTS.iter().map(|argument| (*argument).to_owned()),
        )
        .expect("every argument is given");

        assert_eq!(
            [
                &inputs.compile_commands_file.sys_dir,
                &inputs.compile_commands_file.output_path,
                &inputs.response_file.sys_dir,
                &inputs.response_file.output_path,
            ],
            [
                &PathBuf::from(ARGUMENTS[0]),
                &PathBuf::from(ARGUMENTS[1]),
                &PathBuf::from(ARGUMENTS[0]),
                &PathBuf::from(ARGUMENTS[2]),
            ]
        );
    }
}
