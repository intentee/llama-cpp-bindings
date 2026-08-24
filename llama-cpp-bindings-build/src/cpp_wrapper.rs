use std::path::Path;

use crate::BuildError;
use crate::native_sources::WRAPPER_SOURCES;
use crate::target_os::TargetOs;

pub fn compile_cpp_wrappers(llama_src: &Path, target_os: &TargetOs) -> Result<(), BuildError> {
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .include(".")
        .include("GSL/include")
        .include(llama_src)
        .include(llama_src.join("common"))
        .include(llama_src.join("include"))
        .include(llama_src.join("ggml/include"))
        .include(llama_src.join("vendor"))
        .flag_if_supported("-std=c++17")
        .pic(true);

    for source in WRAPPER_SOURCES {
        build.file(source);
    }

    if target_os.is_msvc() {
        build.flag("/std:c++17");
        build.flag("/EHsc");
    }

    if target_os.is_android() && cfg!(feature = "static-stdcxx") {
        build.cpp_link_stdlib(None);
    }

    build
        .try_compile("llama_cpp_bindings_sys_common_wrapper")
        .map_err(BuildError::NativeWrapper)
}
