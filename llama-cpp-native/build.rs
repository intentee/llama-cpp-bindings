use std::io;
use std::process::ExitCode;

use llama_cpp_bindings_build::build_script::BuildScript;
use llama_cpp_bindings_build::native_library_build::NativeLibraryBuild;

fn main() -> Result<ExitCode, io::Error> {
    BuildScript::run_in_process::<NativeLibraryBuild>()
}
