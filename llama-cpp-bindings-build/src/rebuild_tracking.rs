use std::path::Path;

use crate::native_sources::{WRAPPER_HEADERS, WRAPPER_SOURCES};

pub fn register_rebuild_triggers(llama_src: &Path) {
    println!("cargo:rerun-if-changed=build.rs");

    for path in WRAPPER_HEADERS.iter().chain(WRAPPER_SOURCES) {
        println!("cargo:rerun-if-changed={path}");
    }

    println!("cargo:rerun-if-changed={}", llama_src.display());
}
