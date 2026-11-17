use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::build_error::BuildError;
use llama_cpp_bindings_build::native_build::NativeBuild;
use llama_cpp_error_chain::error_chain::error_chain;

fn build() -> Result<(), BuildError> {
    NativeBuild::from_environment(&BuildEnvironment::of_process())?.run(&mut io::stdout())
}

fn main() -> ExitCode {
    match build() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {}", error_chain(&error));

            ExitCode::FAILURE
        }
    }
}
