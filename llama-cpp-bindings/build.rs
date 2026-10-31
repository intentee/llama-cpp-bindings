use std::env;

fn main() {
    if let Ok(dir) = env::var("DEP_LLAMA_BACKENDS_DIR") {
        println!("cargo:rustc-env=GGML_BACKENDS_DIR={dir}");
    }
}
