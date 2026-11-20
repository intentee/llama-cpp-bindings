mod sys_build_error;

use std::io;

use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::native_build::NativeBuild;

use crate::sys_build_error::SysBuildError;

fn main() -> Result<(), SysBuildError> {
    NativeBuild::from_environment(&BuildEnvironment::of_process())
        .and_then(|native_build| native_build.run(&mut io::stdout()))
        .map_err(SysBuildError::NativeBuild)
}
