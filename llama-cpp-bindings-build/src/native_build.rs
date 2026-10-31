use std::io::Write;

use crate::bindings_generation::BindingsGeneration;
use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::cmake_build::CmakeBuild;
use crate::cmake_configuration::CmakeConfiguration;
use crate::emit_directives::emit_directives;
use crate::link_directives::link_directives;
use crate::rebuild_triggers::rebuild_triggers;
use crate::wrapper_compilation::WrapperCompilation;

/// Every step of the native build, decided before any of them runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBuild {
    pub bindings: BindingsGeneration,
    pub wrappers: WrapperCompilation,
    pub cmake: CmakeBuild,
    pub directives: Vec<CargoDirective>,
    pub include_debug_messages: bool,
}

impl NativeBuild {
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`CmakeConfiguration::of`] or [`link_directives`].
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let CmakeConfiguration {
            defines,
            compiler_flags,
            required_directories,
            directives: cmake_directives,
        } = CmakeConfiguration::of(context, environment)?;

        Ok(Self {
            bindings: BindingsGeneration::of(context, environment),
            wrappers: WrapperCompilation::of(context, environment),
            cmake: CmakeBuild {
                source_dir: context.llama_src(),
                out_dir: context.cmake_dir(),
                profile: context.profile.clone(),
                target_triple: context.target_triple.clone(),
                host_triple: context.host_triple.clone(),
                defines,
                compiler_flags,
                required_directories,
            },
            directives: [
                context.debug_messages(),
                rebuild_triggers(context),
                cmake_directives,
                link_directives(context, environment)?,
            ]
            .concat(),
            include_debug_messages: context.include_debug_messages,
        })
    }

    /// Generates the bindings, compiles the wrappers, builds llama.cpp, and only then writes the
    /// cargo directives to `output`.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of the first step that fails.
    pub fn run(self, output: &mut impl Write) -> Result<(), BuildError> {
        let bindings_directives = self.bindings.generate()?;

        self.wrappers.compile()?;
        self.cmake.build()?;

        emit_directives(
            &[self.directives, bindings_directives].concat(),
            self.include_debug_messages,
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;

    use super::NativeBuild;
    use crate::bindings_generation::BindingsGeneration;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_build::CmakeBuild;
    use crate::cmake_define::CmakeDefine;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;
    use crate::wrapper_compilation::WrapperCompilation;

    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    const HOST_TRIPLE: &str = "x86_64-unknown-linux-gnu";
    #[cfg(all(target_arch = "aarch64", target_os = "linux", target_env = "gnu"))]
    const HOST_TRIPLE: &str = "aarch64-unknown-linux-gnu";
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    const HOST_TRIPLE: &str = "x86_64-apple-darwin";
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    const HOST_TRIPLE: &str = "aarch64-apple-darwin";

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

    const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    fn fixture_build(root: &Path) -> NativeBuild {
        NativeBuild {
            bindings: BindingsGeneration {
                header: PathBuf::from(FIXTURES).join("bindings/wrapper.h"),
                clang_args: Vec::new(),
                output: root.join("bindings.rs"),
            },
            wrappers: WrapperCompilation {
                sources: vec![PathBuf::from(FIXTURES).join("wrapper/wrapper_fixture.cpp")],
                include_dirs: vec![PathBuf::from(FIXTURES).join("wrapper")],
                flags: vec![format!("-isystem{FIXTURES}/wrapper/vendor")],
                cpp_standard_flag: "-std=c++17".to_owned(),
                defines: vec!["WRAPPER_FIXTURE_DEFINE"],
                cpp_stdlib_linked_separately: false,
                target_triple: HOST_TRIPLE.to_owned(),
                host_triple: HOST_TRIPLE.to_owned(),
                opt_level: "0".to_owned(),
                debug_info: "false".to_owned(),
                out_dir: root.join("wrapper"),
            },
            cmake: CmakeBuild {
                source_dir: PathBuf::from(FIXTURES).join("cmake_project"),
                out_dir: root.join("cmake"),
                profile: "Release".to_owned(),
                target_triple: HOST_TRIPLE.to_owned(),
                host_triple: HOST_TRIPLE.to_owned(),
                defines: vec![CmakeDefine::new("FIXTURE_LIBRARY_NAME", "fixture")],
                compiler_flags: vec!["-DFIXTURE_VALUE=1".to_owned()],
                required_directories: vec![root.join("cmake/backends")],
            },
            directives: vec![
                CargoDirective::LinkSearchNative(root.join("cmake/lib")),
                CargoDirective::LinkLibrary(NativeLibrary {
                    link_kind: LinkKind::Static,
                    name: "fixture",
                }),
                CargoDirective::Debug("fixture planned".to_owned()),
            ],
            include_debug_messages: false,
        }
    }

    fn build_root(name: &str) -> PathBuf {
        let root =
            env::temp_dir().join(format!("llama-cpp-bindings-build-{name}-{}", process::id()));

        fs::create_dir_all(&root).unwrap();

        root
    }

    #[test]
    fn a_successful_build_writes_every_artifact_whoever_links_the_cxx_runtime_then_the_directives()
    {
        for cpp_stdlib_linked_separately in [false, true] {
            let root = build_root(&format!("successful-{cpp_stdlib_linked_separately}"));
            let mut native_build = fixture_build(&root);
            let mut output = Vec::new();

            native_build.wrappers.cpp_stdlib_linked_separately = cpp_stdlib_linked_separately;
            native_build.run(&mut output).unwrap();

            assert_eq!(
                String::from_utf8(output).unwrap(),
                format!(
                    "cargo:rustc-link-search=native={root}/cmake/lib\n\
                     cargo:rustc-link-lib=static=fixture\n\
                     cargo:rerun-if-env-changed=TARGET\n\
                     cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS\n\
                     cargo:rerun-if-changed={FIXTURES}/bindings/wrapper.h\n\
                     cargo:rerun-if-changed={FIXTURES}/bindings/llama_types.h\n",
                    root = root.display()
                )
            );
            assert_eq!(
                [
                    root.join("bindings.rs").is_file(),
                    root.join("wrapper/libllama_cpp_bindings_sys_common_wrapper.a")
                        .is_file(),
                    root.join("cmake/lib/libfixture.a").is_file(),
                    root.join("cmake/backends").is_dir(),
                ],
                [true; 4]
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn a_failed_wrapper_compilation_stops_the_build_before_cmake_and_any_directive() {
        let root = build_root("failed-wrappers");
        let mut native_build = fixture_build(&root);
        let mut output = Vec::new();

        native_build.wrappers.out_dir = PathBuf::from(MANIFEST);

        assert_eq!(
            native_build
                .run(&mut output)
                .err()
                .map(|error| error.to_string()),
            Some(
                "native wrapper compilation failed: IOError: File exists (os error 17)".to_owned()
            )
        );
        assert_eq!(
            [output.is_empty(), root.join("cmake").exists()],
            [true, false]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_cmake_build_that_cannot_create_its_directories_emits_no_directive() {
        let root = build_root("failed-cmake");
        let mut native_build = fixture_build(&root);
        let mut output = Vec::new();

        native_build.cmake.required_directories = vec![PathBuf::from(MANIFEST).join("backends")];

        assert_eq!(
            native_build
                .run(&mut output)
                .err()
                .map(|error| error.to_string()),
            Some(format!(
                "filesystem operation failed for {MANIFEST}/backends: Not a directory (os error \
                 20)"
            ))
        );
        assert_eq!(output, Vec::<u8>::new());
        fs::remove_dir_all(root).unwrap();
    }
}
