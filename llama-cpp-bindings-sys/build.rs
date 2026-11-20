use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::build_script::BuildScript;

fn main() -> Result<ExitCode, io::Error> {
    BuildScript {
        directives: io::stdout(),
        diagnostics: io::stderr(),
    }
    .run(&BuildEnvironment::of_process())
}
