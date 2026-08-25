use std::fs::File;
use std::path::Path;
use std::path::PathBuf;

use crate::compile_command::CompileCommand;
use crate::cpp_standard::CPP_STANDARD;
use crate::wrapper_include_dirs::WRAPPER_INCLUDE_DIRS;
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
                let mut arguments = vec![CPP_COMPILER.to_owned(), format!("-std={CPP_STANDARD}")];

                for include_dir in WRAPPER_INCLUDE_DIRS {
                    arguments.push(format!("-I{}", absolute_path(&self.sys_dir, include_dir)));
                }

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

    use super::CompileCommandsFile;
    use crate::wrapper_include_dirs::WRAPPER_INCLUDE_DIRS;
    use crate::wrapper_source_paths::wrapper_source_paths;
    use crate::wrapper_sources_error::WrapperSourcesError;

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
            commands
                .first()
                .map(|command| command
                    .arguments
                    .iter()
                    .filter(|argument| argument.starts_with("-I"))
                    .cloned()
                    .collect::<Vec<String>>())
                .unwrap_or_default(),
            WRAPPER_INCLUDE_DIRS
                .iter()
                .map(|include_dir| format!("-I/repo/llama-cpp-bindings-sys/{include_dir}"))
                .collect::<Vec<String>>()
        );
    }

    #[test]
    fn the_compiler_and_standard_lead_the_command_line() {
        let commands = database_for("/unused").commands();

        assert_eq!(
            commands
                .first()
                .map(|command| command
                    .arguments
                    .iter()
                    .take(2)
                    .cloned()
                    .collect::<Vec<String>>())
                .unwrap_or_default(),
            vec!["c++".to_owned(), "-std=c++17".to_owned()]
        );
    }

    #[test]
    fn creating_the_database_under_a_missing_directory_reports_the_path() {
        let output_path = "/nonexistent-directory/compile_commands.json";

        assert!(matches!(
            database_for(output_path).write(),
            Err(WrapperSourcesError::Create { ref path, .. }) if path == Path::new(output_path)
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_destination_that_cannot_absorb_the_bytes_reports_the_path() {
        assert!(matches!(
            database_for("/dev/full").write(),
            Err(WrapperSourcesError::Serialize { ref path, .. }) if path == Path::new("/dev/full")
        ));
    }
}
