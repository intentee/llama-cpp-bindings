use std::env;
use std::env::VarError;

fn main() -> Result<(), VarError> {
    env::var("TARGET").map(|target| {
        println!("cargo:rustc-env=LLAMA_CPP_BINDINGS_BUILD_TARGET={target}");
        println!("cargo:rerun-if-changed=build.rs");
    })
}
