use std::env;
use std::path::PathBuf;

use llama_cpp_wrapper_sources::compile_commands_file::CompileCommandsFile;
use llama_cpp_wrapper_sources::wrapper_sources_error::WrapperSourcesError;
use llama_cpp_wrapper_sources::wrapper_sources_response_file::WrapperSourcesResponseFile;

fn main() -> Result<(), WrapperSourcesError> {
    let mut arguments = env::args().skip(1);

    let sys_dir = arguments
        .next()
        .ok_or(WrapperSourcesError::MissingArgument { name: "sys-dir" })?;
    let database_path = arguments
        .next()
        .ok_or(WrapperSourcesError::MissingArgument {
            name: "compile-commands-path",
        })?;
    let response_file_path = arguments
        .next()
        .ok_or(WrapperSourcesError::MissingArgument {
            name: "response-file-path",
        })?;

    CompileCommandsFile {
        output_path: PathBuf::from(database_path),
        sys_dir: PathBuf::from(&sys_dir),
    }
    .write()?;

    WrapperSourcesResponseFile {
        output_path: PathBuf::from(response_file_path),
        sys_dir: PathBuf::from(&sys_dir),
    }
    .write()
}
