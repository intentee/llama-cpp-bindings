use std::path::Path;
use std::path::PathBuf;

use llama_cpp_wrapper_sources::constants::CPP_STANDARD;
use llama_cpp_wrapper_sources::constants::FAULT_INJECTION_DEFINE;
use llama_cpp_wrapper_sources::constants::WRAPPER_INCLUDE_DIRS;
use llama_cpp_wrapper_sources::system_include_flags::system_include_flags;
use llama_cpp_wrapper_sources::wrapper_source_paths::wrapper_source_paths;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::target_os::TargetOs;

const WRAPPER_LIBRARY: &str = "llama_cpp_bindings_sys_common_wrapper";

/// Compiles the C++ wrapper sources into the static wrapper library in `out_dir`, against the
/// llama.cpp headers of the native package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WrapperCompilation {
    pub sources: Vec<PathBuf>,
    pub include_dirs: Vec<PathBuf>,
    pub flags: Vec<String>,
    pub cpp_standard: &'static str,
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
    pub fn of(
        BuildContext {
            manifest_dir,
            target_os,
            target_triple,
            host_triple,
            opt_level,
            debug_info,
            out_dir,
            ..
        }: &BuildContext,
        environment: &BuildEnvironment,
        llama_cpp_dir: &Path,
    ) -> Self {
        Self {
            sources: wrapper_source_paths(manifest_dir),
            include_dirs: WRAPPER_INCLUDE_DIRS
                .iter()
                .map(|include_dir| manifest_dir.join(include_dir))
                .collect(),
            flags: system_include_flags(manifest_dir, llama_cpp_dir),
            cpp_standard: CPP_STANDARD,
            defines: if environment.has(BuildFeature::FaultInjection) {
                vec![FAULT_INJECTION_DEFINE]
            } else {
                Vec::new()
            },
            cpp_stdlib_linked_separately: matches!(target_os, TargetOs::Android(_))
                && environment.has(BuildFeature::StaticStdcxx),
            target_triple: target_triple.clone(),
            host_triple: host_triple.clone(),
            opt_level: opt_level.clone(),
            debug_info: debug_info.clone(),
            out_dir: out_dir.clone(),
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

        build.std(self.cpp_standard);

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

    use llama_cpp_wrapper_sources::constants::FAULT_INJECTION_DEFINE;
    use llama_cpp_wrapper_sources::system_include_flags::system_include_flags;
    use llama_cpp_wrapper_sources::wrapper_source_paths::wrapper_source_paths;

    use super::WrapperCompilation;
    use crate::build_context::BuildContext;
    use crate::build_context::tests::android_overrides;
    use crate::build_context::tests::cargo_environment;
    use crate::build_environment::tests::EnvironmentVariable;
    use crate::build_feature::BuildFeature;

    fn planned(features: &[BuildFeature], overrides: &[EnvironmentVariable]) -> WrapperCompilation {
        let environment = cargo_environment(features, overrides);

        WrapperCompilation::of(
            &BuildContext::of_linux_host(&environment),
            &environment,
            Path::new("/native/llama.cpp"),
        )
    }

    fn linux_compilation() -> WrapperCompilation {
        WrapperCompilation {
            sources: wrapper_source_paths(Path::new("/sys")),
            include_dirs: vec![PathBuf::from("/sys/.")],
            flags: system_include_flags(Path::new("/sys"), Path::new("/native/llama.cpp")),
            cpp_standard: "c++17",
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
        assert_eq!(planned(&[], &[]), linux_compilation());
    }

    #[test]
    fn fault_injection_defines_its_switch() {
        assert_eq!(
            planned(&[BuildFeature::FaultInjection], &[]),
            WrapperCompilation {
                defines: vec![FAULT_INJECTION_DEFINE],
                ..linux_compilation()
            }
        );
    }

    #[test]
    fn a_static_android_stl_is_linked_separately() {
        assert_eq!(
            planned(&[BuildFeature::StaticStdcxx], &android_overrides()),
            WrapperCompilation {
                cpp_stdlib_linked_separately: true,
                target_triple: "aarch64-linux-android".to_owned(),
                ..linux_compilation()
            }
        );
    }
}
