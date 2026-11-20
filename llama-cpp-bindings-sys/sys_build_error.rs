use llama_cpp_bindings_build::build_error::BuildError;

#[derive(Debug, thiserror::Error)]
pub enum SysBuildError {
    #[error("building llama.cpp and its bindings failed")]
    NativeBuild(#[source] BuildError),
}
