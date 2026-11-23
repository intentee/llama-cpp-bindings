use std::io::Write;
use std::iter;
use std::path::PathBuf;

use llama_cpp_wrapper_sources::constants::WRAPPER_HEADERS;
use llama_cpp_wrapper_sources::constants::WRAPPER_SOURCES;

use crate::bindings_generation::BindingsGeneration;
use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::constants::LLAMA_CPP_ROOT_VARIABLE;
use crate::emit_directives::emit_directives;
use crate::planned_build::PlannedBuild;
use crate::wrapper_compilation::WrapperCompilation;

/// Every step of generating the bindings and compiling the C++ wrapper of the sys package against
/// the llama.cpp headers of the native package, decided before any of them runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingsBuild {
    pub bindings: BindingsGeneration,
    pub wrappers: WrapperCompilation,
    pub directives: Vec<CargoDirective>,
    pub include_debug_messages: bool,
}

impl PlannedBuild for BindingsBuild {
    /// Plans the bindings and the wrapper against the llama.cpp directory the native package
    /// hands to the build script, rebuilding them when the build script or a wrapper file
    /// changes.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when cargo did not hand the build script the
    /// llama.cpp directory of the native package.
    fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let llama_cpp_dir = PathBuf::from(environment.required(LLAMA_CPP_ROOT_VARIABLE)?);

        Ok(Self {
            bindings: BindingsGeneration::of(context, environment, &llama_cpp_dir),
            wrappers: WrapperCompilation::of(context, environment, &llama_cpp_dir),
            directives: iter::once(context.build_script_trigger())
                .chain(
                    WRAPPER_HEADERS
                        .iter()
                        .chain(WRAPPER_SOURCES)
                        .map(|wrapper_file| {
                            CargoDirective::RerunIfChanged(context.manifest_dir.join(wrapper_file))
                        }),
                )
                .collect(),
            include_debug_messages: context.include_debug_messages,
        })
    }

    /// Generates the bindings, compiles the wrapper, and only then writes the cargo directives to
    /// `output`.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of the first step that fails.
    fn run(self, output: &mut impl Write) -> Result<(), BuildError> {
        let bindings_directives = self.bindings.generate()?;

        self.wrappers.compile()?;

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
    use std::iter;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;

    use enumflags2::BitFlags;

    use llama_cpp_error_chain::error_chain::error_chain;
    use llama_cpp_wrapper_sources::constants::WRAPPER_HEADERS;
    use llama_cpp_wrapper_sources::constants::WRAPPER_SOURCES;

    use super::BindingsBuild;
    use crate::bindings_generation::BindingsGeneration;
    use crate::build_context::BuildContext;
    use crate::build_context::tests::cargo_environment;
    use crate::build_environment::BuildEnvironment;
    use crate::build_environment::tests::EnvironmentVariable;
    use crate::build_environment::tests::variable;
    use crate::cargo_directive::CargoDirective;
    use crate::planned_build::PlannedBuild;
    use crate::wrapper_compilation::WrapperCompilation;

    const HOST_TRIPLE: &str = include_str!(concat!(env!("OUT_DIR"), "/host_triple"));

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

    const FIXTURE_CMAKE_PROJECT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cmake_project");

    const MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");

    fn native_llama_cpp_dir() -> EnvironmentVariable {
        variable("DEP_LLAMA_ROOT", "/native/llama.cpp")
    }

    fn fixture_build(root: &Path) -> BindingsBuild {
        BindingsBuild {
            bindings: BindingsGeneration {
                header: PathBuf::from(FIXTURES).join("bindings/wrapper.h"),
                clang_args: Vec::new(),
                output: root.join("bindings.rs"),
            },
            wrappers: WrapperCompilation {
                sources: vec![PathBuf::from(FIXTURES).join("wrapper/wrapper_fixture.cpp")],
                include_dirs: vec![PathBuf::from(FIXTURES).join("wrapper")],
                flags: vec![format!("-isystem{FIXTURES}/wrapper/vendor")],
                cpp_standard: "c++17",
                defines: vec!["WRAPPER_FIXTURE_DEFINE"],
                cpp_stdlib_linked_separately: false,
                target_triple: HOST_TRIPLE.to_owned(),
                host_triple: HOST_TRIPLE.to_owned(),
                opt_level: "0".to_owned(),
                debug_info: "false".to_owned(),
                out_dir: root.join("wrapper"),
            },
            directives: vec![CargoDirective::Debug("fixture planned".to_owned())],
            include_debug_messages: false,
        }
    }

    fn build_root(name: &str) -> PathBuf {
        let root =
            env::temp_dir().join(format!("llama-cpp-bindings-build-{name}-{}", process::id()));

        fs::create_dir_all(&root).unwrap();

        root
    }

    fn failure(environment: &BuildEnvironment) -> [Option<String>; 2] {
        let mut output = Vec::new();

        [
            BindingsBuild::from_environment(environment)
                .and_then(|bindings_build| bindings_build.run(&mut output))
                .err()
                .map(|error| error_chain(&error)),
            String::from_utf8(output).ok(),
        ]
    }

    #[test]
    fn the_bindings_are_generated_against_the_native_llama_cpp_and_rebuilt_with_the_wrappers() {
        let environment = cargo_environment(BitFlags::empty(), &[native_llama_cpp_dir()]);
        let context = BuildContext::of_linux_host(&environment);

        assert_eq!(
            BindingsBuild::of(&context, &environment).unwrap(),
            BindingsBuild {
                bindings: BindingsGeneration::of(
                    &context,
                    &environment,
                    Path::new("/native/llama.cpp")
                ),
                wrappers: WrapperCompilation::of(
                    &context,
                    &environment,
                    Path::new("/native/llama.cpp")
                ),
                directives: iter::once(CargoDirective::RerunIfChanged(PathBuf::from(
                    "/sys/build.rs"
                )))
                .chain(
                    WRAPPER_HEADERS
                        .iter()
                        .chain(WRAPPER_SOURCES)
                        .map(|wrapper_file| {
                            CargoDirective::RerunIfChanged(PathBuf::from(format!(
                                "/sys/{wrapper_file}"
                            )))
                        })
                )
                .collect(),
                include_debug_messages: false,
            }
        );
    }

    #[test]
    fn a_successful_build_writes_every_artifact_whoever_links_the_cxx_runtime_then_the_directives()
    {
        for cpp_stdlib_linked_separately in [false, true] {
            let root = build_root(&format!("successful-{cpp_stdlib_linked_separately}"));
            let mut bindings_build = fixture_build(&root);
            let mut output = Vec::new();

            bindings_build.wrappers.cpp_stdlib_linked_separately = cpp_stdlib_linked_separately;
            bindings_build.run(&mut output).unwrap();

            assert_eq!(
                String::from_utf8(output).unwrap(),
                format!(
                    "cargo:rerun-if-env-changed=TARGET\n\
                     cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS\n\
                     cargo:rerun-if-changed={FIXTURES}/bindings/wrapper.h\n\
                     cargo:rerun-if-changed={FIXTURES}/bindings/llama_types.h\n"
                )
            );
            assert_eq!(
                [
                    root.join("bindings.rs").is_file(),
                    root.join("wrapper/libllama_cpp_bindings_sys_common_wrapper.a")
                        .is_file(),
                ],
                [true; 2]
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn a_failed_wrapper_compilation_emits_no_directive() {
        let root = build_root("failed-wrappers");
        let mut bindings_build = fixture_build(&root);
        let mut output = Vec::new();

        bindings_build.wrappers.out_dir = PathBuf::from(MANIFEST);

        assert_eq!(
            bindings_build
                .run(&mut output)
                .err()
                .map(|error| error_chain(&error)),
            Some(
                "native wrapper compilation failed: IOError: File exists (os error 17)".to_owned()
            )
        );
        assert_eq!(output, Vec::<u8>::new());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failure_at_any_planning_stage_is_reported_and_nothing_is_emitted() {
        assert_eq!(
            [
                failure(&cargo_environment(BitFlags::empty(), &[])),
                failure(&cargo_environment(
                    BitFlags::empty(),
                    &[
                        native_llama_cpp_dir(),
                        variable("CARGO_MANIFEST_DIR", FIXTURE_CMAKE_PROJECT)
                    ]
                )),
            ],
            [
                [
                    Some(
                        "environment variable DEP_LLAMA_ROOT could not be read: environment \
                         variable not found"
                            .to_owned()
                    ),
                    Some(String::new())
                ],
                [
                    Some(format!(
                        "bindgen failed: header '{FIXTURE_CMAKE_PROJECT}/wrapper.h' does not exist."
                    )),
                    Some(String::new())
                ],
            ]
        );
    }
}
