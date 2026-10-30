use llama_cpp_bindings_build::build_environment::BuildEnvironment;
use llama_cpp_bindings_build::build_error::BuildError;

fn main() -> Result<(), BuildError> {
    llama_cpp_bindings_build::build(&BuildEnvironment::of_process(), &mut std::io::stdout())
}
