use std::env;
use std::process::ExitCode;

use llama_cpp_error_chain::error_chain::error_chain;
use llama_cpp_wrapper_sources::wrapper_build_inputs::WrapperBuildInputs;

fn main() -> ExitCode {
    let emitted = WrapperBuildInputs::from_arguments(env::args().skip(1))
        .and_then(|build_inputs| build_inputs.write());

    match emitted {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {}", error_chain(&error));

            ExitCode::FAILURE
        }
    }
}
