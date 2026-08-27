use llama_cpp_wrapper_sources::cpp_standard::CPP_STANDARD;
use llama_cpp_wrapper_sources::wrapper_include_dirs::WRAPPER_INCLUDE_DIRS;
use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

use crate::BuildError;
use crate::target_os::TargetOs;

pub fn compile_cpp_wrappers(target_os: TargetOs) -> Result<(), BuildError> {
    let mut build = cc::Build::new();

    build.cpp(true).pic(true);

    for include_dir in WRAPPER_INCLUDE_DIRS {
        build.include(include_dir);
    }

    build.flag_if_supported(format!("-std={CPP_STANDARD}"));

    for source in WRAPPER_SOURCES {
        build.file(source);
    }

    if target_os.is_msvc() {
        build.flag(format!("/std:{CPP_STANDARD}"));
        build.flag("/EHsc");
    }

    if target_os.is_android() && cfg!(feature = "static-stdcxx") {
        build.cpp_link_stdlib(None);
    }

    build
        .try_compile("llama_cpp_bindings_sys_common_wrapper")
        .map_err(BuildError::NativeWrapper)
}
