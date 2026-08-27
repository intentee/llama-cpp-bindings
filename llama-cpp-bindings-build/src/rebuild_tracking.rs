use std::path::Path;

use llama_cpp_wrapper_sources::wrapper_headers::WRAPPER_HEADERS;
use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

pub fn register_rebuild_triggers(llama_src: &Path) {
    println!("cargo:rerun-if-changed=build.rs");

    for path in WRAPPER_HEADERS.iter().chain(WRAPPER_SOURCES) {
        println!("cargo:rerun-if-changed={path}");
    }

    println!("cargo:rerun-if-changed={}", llama_src.display());
}
