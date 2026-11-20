use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use crate::compile_command::CompileCommand;
use crate::constants::CPP_STANDARD;
use crate::constants::FAULT_INJECTION_DEFINE;
use crate::constants::WRAPPER_INCLUDE_DIRS;
use crate::system_include_flags::system_include_flags;
use crate::wrapper_source_paths::wrapper_source_paths;
use crate::wrapper_sources_error::WrapperSourcesError;

const CPP_COMPILER: &str = "c++";

fn absolute_path(directory: &Path, relative: &str) -> String {
    directory.join(relative).display().to_string()
}

pub struct CompileCommandsFile {
    pub output_path: PathBuf,
    pub sys_dir: PathBuf,
}

impl CompileCommandsFile {
    #[must_use]
    pub fn commands(&self) -> Vec<CompileCommand> {
        wrapper_source_paths(&self.sys_dir)
            .into_iter()
            .map(|file| {
                let mut arguments = vec![
                    CPP_COMPILER.to_owned(),
                    format!("-std={CPP_STANDARD}"),
                    format!("-D{FAULT_INJECTION_DEFINE}"),
                ];

                for include_dir in WRAPPER_INCLUDE_DIRS {
                    arguments.push(format!("-I{}", absolute_path(&self.sys_dir, include_dir)));
                }

                arguments.extend(system_include_flags(&self.sys_dir));

                arguments.push("-c".to_owned());
                arguments.push(file.display().to_string());

                CompileCommand {
                    arguments,
                    directory: self.sys_dir.clone(),
                    file,
                }
            })
            .collect()
    }

    /// # Errors
    ///
    /// Returns [`WrapperSourcesError`] when the destination file cannot be created or
    /// the compilation database cannot be written to it.
    pub fn write(&self) -> Result<(), WrapperSourcesError> {
        let destination =
            File::create(&self.output_path).map_err(|source| WrapperSourcesError::Create {
                path: self.output_path.clone(),
                source,
            })?;

        serde_json::to_writer_pretty(destination, &self.commands()).map_err(|source| {
            WrapperSourcesError::Serialize {
                path: self.output_path.clone(),
                source,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use llama_cpp_error_chain::error_chain::error_chain;

    use super::CompileCommandsFile;
    use crate::constants::WRAPPER_INCLUDE_DIRS;
    use crate::system_include_flags::system_include_flags;
    use crate::wrapper_source_paths::wrapper_source_paths;

    fn database_for(output_path: &str) -> CompileCommandsFile {
        CompileCommandsFile {
            output_path: PathBuf::from(output_path),
            sys_dir: PathBuf::from("/repo/llama-cpp-bindings-sys"),
        }
    }

    #[test]
    fn every_wrapper_source_becomes_one_entry() {
        let commands = database_for("/unused").commands();

        assert_eq!(
            commands
                .iter()
                .map(|command| command.file.clone())
                .collect::<Vec<PathBuf>>(),
            wrapper_source_paths(Path::new("/repo/llama-cpp-bindings-sys"))
        );
    }

    #[test]
    fn every_include_dir_is_resolved_against_the_sys_dir() {
        let commands = database_for("/unused").commands();

        assert_eq!(
            commands.first().map_or_default(|command| command
                .arguments
                .iter()
                .filter(|argument| argument.starts_with("-I"))
                .cloned()
                .collect::<Vec<String>>()),
            WRAPPER_INCLUDE_DIRS
                .iter()
                .map(|include_dir| {
                    format!(
                        "-I{}",
                        Path::new("/repo/llama-cpp-bindings-sys")
                            .join(include_dir)
                            .display()
                    )
                })
                .collect::<Vec<String>>()
        );
    }

    #[test]
    fn every_command_includes_the_vendored_headers_as_system_headers() {
        let commands = database_for("/unused").commands();

        assert_eq!(
            commands
                .iter()
                .map(|command| command
                    .arguments
                    .iter()
                    .filter(|argument| argument.starts_with("-isystem"))
                    .cloned()
                    .collect::<Vec<String>>())
                .collect::<Vec<Vec<String>>>(),
            vec![
                system_include_flags(Path::new("/repo/llama-cpp-bindings-sys"));
                wrapper_source_paths(Path::new("/repo/llama-cpp-bindings-sys")).len()
            ]
        );
    }

    #[test]
    fn the_compiler_standard_and_fault_injection_lead_the_command_line() {
        let commands = database_for("/unused").commands();

        assert_eq!(
            commands.first().map_or_default(|command| command
                .arguments
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<String>>()),
            vec![
                "c++".to_owned(),
                "-std=c++17".to_owned(),
                "-DLLAMA_RS_FAULT_INJECTION".to_owned()
            ]
        );
    }

    #[test]
    fn creating_the_database_under_a_missing_directory_reports_the_path() {
        assert_eq!(
            database_for("/nonexistent-directory/compile_commands.json")
                .write()
                .err()
                .map(|error| error_chain(&error)),
            Some(
                "compilation database file /nonexistent-directory/compile_commands.json could \
                 not be created: No such file or directory (os error 2)"
                    .to_owned()
            )
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_destination_that_cannot_absorb_the_bytes_reports_the_path() {
        assert_eq!(
            database_for("/dev/full")
                .write()
                .err()
                .map(|error| error_chain(&error)),
            Some(
                "compilation database could not be written to /dev/full: No space left on \
                 device (os error 28)"
                    .to_owned()
            )
        );
    }
}
