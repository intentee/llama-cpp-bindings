use std::path::PathBuf;

use llama_cpp_wrapper_sources::cpp_standard::CPP_STANDARD;
use llama_cpp_wrapper_sources::fault_injection_define::FAULT_INJECTION_DEFINE;
use llama_cpp_wrapper_sources::system_include_flags::system_include_flags;
use llama_cpp_wrapper_sources::wrapper_include_dirs::WRAPPER_INCLUDE_DIRS;
use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::target_os::TargetOs;

const WRAPPER_LIBRARY: &str = "llama_cpp_bindings_sys_common_wrapper";

/// Compiles the C++ wrapper sources into the static wrapper library in `out_dir`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrapperCompilation {
    pub sources: Vec<PathBuf>,
    pub include_dirs: Vec<PathBuf>,
    pub flags: Vec<String>,
    pub cpp_standard_flag: String,
    pub defines: Vec<&'static str>,
    pub cpp_stdlib_linked_separately: bool,
    pub target_triple: String,
    pub host_triple: String,
    pub opt_level: String,
    pub debug_info: String,
    pub out_dir: PathBuf,
}

impl WrapperCompilation {
    #[must_use]
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Self {
        Self {
            sources: WRAPPER_SOURCES
                .iter()
                .map(|source| context.sys_dir.join(source))
                .collect(),
            include_dirs: WRAPPER_INCLUDE_DIRS
                .iter()
                .map(|include_dir| context.sys_dir.join(include_dir))
                .collect(),
            flags: system_include_flags(&context.sys_dir),
            cpp_standard_flag: format!("-std={CPP_STANDARD}"),
            defines: if environment.has(BuildFeature::FaultInjection) {
                vec![FAULT_INJECTION_DEFINE]
            } else {
                Vec::new()
            },
            cpp_stdlib_linked_separately: matches!(context.target_os, TargetOs::Android(_))
                && environment.has(BuildFeature::StaticStdcxx),
            target_triple: context.target_triple.clone(),
            host_triple: context.host_triple.clone(),
            opt_level: context.opt_level.clone(),
            debug_info: context.debug_info.clone(),
            out_dir: context.out_dir.clone(),
        }
    }

    /// # Errors
    ///
    /// Returns [`BuildError::NativeWrapper`] when the wrapper library cannot be compiled.
    pub fn compile(&self) -> Result<(), BuildError> {
        let mut build = cc::Build::new();

        build.cpp(true).pic(true).includes(&self.include_dirs);

        for flag in &self.flags {
            build.flag(flag);
        }

        build.flag_if_supported(&self.cpp_standard_flag);

        for define in &self.defines {
            build.define(define, None);
        }

        build.files(&self.sources);

        if self.cpp_stdlib_linked_separately {
            build.cpp_link_stdlib(None);
        }

        build
            .target(&self.target_triple)
            .host(&self.host_triple)
            .opt_level_str(&self.opt_level)
            .debug_str(&self.debug_info)
            .out_dir(&self.out_dir)
            .try_compile(WRAPPER_LIBRARY)
            .map_err(BuildError::NativeWrapper)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use llama_cpp_wrapper_sources::fault_injection_define::FAULT_INJECTION_DEFINE;
    use llama_cpp_wrapper_sources::system_include_flags::system_include_flags;
    use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

    use super::WrapperCompilation;
    use crate::build_context::tests::ANDROID_OVERRIDES;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_feature::BuildFeature;

    fn planned(features: BitFlags<BuildFeature>, overrides: &[(&str, &str)]) -> WrapperCompilation {
        let environment = cargo_environment(features, overrides);

        WrapperCompilation::of(&context_of(&environment), &environment)
    }

    fn linux_compilation() -> WrapperCompilation {
        WrapperCompilation {
            sources: WRAPPER_SOURCES
                .iter()
                .map(|source| PathBuf::from(format!("/sys/{source}")))
                .collect(),
            include_dirs: vec![PathBuf::from("/sys/.")],
            flags: system_include_flags(Path::new("/sys")),
            cpp_standard_flag: "-std=c++17".to_owned(),
            defines: Vec::new(),
            cpp_stdlib_linked_separately: false,
            target_triple: "x86_64-unknown-linux-gnu".to_owned(),
            host_triple: "x86_64-unknown-linux-gnu".to_owned(),
            opt_level: "3".to_owned(),
            debug_info: "false".to_owned(),
            out_dir: PathBuf::from("/out"),
        }
    }

    #[test]
    fn the_wrappers_compile_as_cxx17_with_the_vendored_headers_as_system_includes() {
        assert_eq!(planned(BitFlags::empty(), &[]), linux_compilation());
    }

    #[test]
    fn fault_injection_defines_its_switch_and_a_static_android_stl_is_linked_separately() {
        assert_eq!(
            planned(
                BuildFeature::FaultInjection | BuildFeature::StaticStdcxx,
                &ANDROID_OVERRIDES
            ),
            WrapperCompilation {
                defines: vec![FAULT_INJECTION_DEFINE],
                cpp_stdlib_linked_separately: true,
                target_triple: "aarch64-linux-android".to_owned(),
                ..linux_compilation()
            }
        );
    }
}
