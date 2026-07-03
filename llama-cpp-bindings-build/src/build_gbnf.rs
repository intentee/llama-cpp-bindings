use std::env;
use std::path::Path;
use std::path::PathBuf;

use crate::target_os::TargetOs;

const LLAMA_BINDINGS_SYS_RELATIVE_PATH: &str = "../llama-cpp-bindings-sys";
const LLAMA_SUBMODULE_RELATIVE_PATH: &str = "../llama-cpp-bindings-sys/llama.cpp";

pub fn build_gbnf() {
    let manifest_dir =
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR env var is required");
    let target_triple = env::var("TARGET").expect("TARGET env var is required in build scripts");
    let target_os = TargetOs::from_target_triple(&target_triple)
        .unwrap_or_else(|error| panic!("Failed to parse target OS: {error}"));

    let manifest = PathBuf::from(&manifest_dir);
    let llama_src = manifest.join(LLAMA_SUBMODULE_RELATIVE_PATH);
    let bindings_sys = manifest.join(LLAMA_BINDINGS_SYS_RELATIVE_PATH);
    let grammar_source_dir = llama_src.join("src");

    assert!(
        grammar_source_dir.join("llama-grammar.h").is_file(),
        "grammar headers not found under {}; ensure the llama.cpp submodule is checked out",
        grammar_source_dir.display()
    );

    register_rebuild_triggers(&manifest);

    compile_wrapper(
        &manifest,
        &llama_src,
        &bindings_sys,
        &grammar_source_dir,
        &target_os,
    );
}

fn compile_wrapper(
    manifest: &Path,
    llama_src: &Path,
    bindings_sys: &Path,
    grammar_source_dir: &Path,
    target_os: &TargetOs,
) {
    let fault_injection = env::var_os("CARGO_FEATURE_FAULT_INJECTION").is_some();

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .warnings(false)
        .include(manifest)
        .include(bindings_sys)
        .include(llama_src.join("include"))
        .include(grammar_source_dir)
        .include(llama_src.join("ggml/include"))
        .flag_if_supported("-std=c++17")
        .pic(true)
        .file(manifest.join("wrapper_gbnf.cpp"));

    if fault_injection {
        build.define("GBNF_FAULT_INJECTION", None);
        build.file(manifest.join("alloc_fault.cpp"));
    }

    if target_os.is_msvc() {
        build.flag("/std:c++17");
        build.flag("/EHsc");
    }

    build.compile("llama_cpp_gbnf_wrapper");
}

fn register_rebuild_triggers(manifest: &Path) {
    println!("cargo:rerun-if-changed=build.rs");

    for source in [
        "wrapper_gbnf.cpp",
        "wrapper_gbnf.h",
        "alloc_fault.cpp",
        "alloc_fault.h",
    ] {
        println!("cargo:rerun-if-changed={}", manifest.join(source).display());
    }
}
