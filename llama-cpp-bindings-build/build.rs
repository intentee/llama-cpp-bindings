use std::env;
use std::env::VarError;

fn main() -> Result<(), VarError> {
    println!(
        "cargo:rustc-env=LLAMA_CPP_BINDINGS_BUILD_TARGET={}",
        env::var("TARGET")?
    );
    println!("cargo:rerun-if-changed=build.rs");

    Ok(())
}
