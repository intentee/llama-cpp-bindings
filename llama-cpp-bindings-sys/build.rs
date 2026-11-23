use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::bindings_build::BindingsBuild;
use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::build_script::BuildScript;

fn main() -> Result<ExitCode, io::Error> {
    BuildScript {
        directives: io::stdout(),
        diagnostics: io::stderr(),
    }
    .run::<BindingsBuild>(&BuildEnvironment::of_process())
}
