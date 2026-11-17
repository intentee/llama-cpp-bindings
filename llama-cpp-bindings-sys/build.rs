use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::native_build::NativeBuild;
use llama_cpp_error_chain::error_chain::error_chain;

fn main() -> ExitCode {
    let built = NativeBuild::from_environment(&BuildEnvironment::of_process())
        .and_then(|native_build| native_build.run(&mut io::stdout()));

    match built {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {}", error_chain(&error));

            ExitCode::FAILURE
        }
    }
}
