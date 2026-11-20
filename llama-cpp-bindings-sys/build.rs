use std::io;

use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::build_error::BuildError;
use llama_cpp_bindings_build::native_build::NativeBuild;

fn main() -> Result<(), BuildError> {
    NativeBuild::from_environment(&BuildEnvironment::of_process())
        .and_then(|native_build| native_build.run(&mut io::stdout()))
}
