use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::bindings_build::BindingsBuild;
use llama_cpp_bindings_build::build_script::BuildScript;

fn main() -> Result<ExitCode, io::Error> {
    BuildScript::run_in_process::<BindingsBuild>()
}
