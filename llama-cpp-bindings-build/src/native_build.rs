use std::io::Write;

use crate::bindings_generation::BindingsGeneration;
use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::cmake_build::CmakeBuild;
use crate::cmake_configuration::CmakeConfiguration;
use crate::cmake_file_api::CmakeFileApi;
use crate::emit_directives::emit_directives;
use crate::host_os::HostOs;
use crate::link_directives::LinkDirectives;
use crate::openmp_linking::OpenmpLinking;
use crate::wrapper_compilation::WrapperCompilation;

/// Every step of the native build, decided before any of them runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeBuild {
    pub bindings: BindingsGeneration,
    pub wrappers: WrapperCompilation,
    pub cmake: CmakeBuild,
    pub directives: Vec<CargoDirective>,
    pub openmp: OpenmpLinking,
    pub include_debug_messages: bool,
}

impl NativeBuild {
    /// Plans the native build of the build script run `environment` describes, on the host the
    /// build script runs on.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`BuildContext::from_environment`] or [`Self::of`].
    pub fn from_environment(environment: &BuildEnvironment) -> Result<Self, BuildError> {
        Self::of(
            &BuildContext::from_environment(environment, HostOs::current())?,
            environment,
        )
    }

    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`CmakeConfiguration::of`] or [`LinkDirectives::of`].
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let CmakeConfiguration {
            defines,
            required_directories,
            directives: cmake_directives,
        } = CmakeConfiguration::of(context, environment)?;

        Ok(Self {
            bindings: BindingsGeneration::of(context, environment),
            wrappers: WrapperCompilation::of(context, environment),
            cmake: CmakeBuild {
                source_dir: context.llama_src(),
                out_dir: context.cmake_dir(),
                file_api: CmakeFileApi {
                    build_dir: context.cmake_build_dir(),
                },
                target_triple: context.target_triple.clone(),
                host_triple: context.host_triple.clone(),
                defines,
                required_directories,
            },
            directives: [
                context.debug_messages(),
                context.rebuild_triggers(),
                cmake_directives,
                LinkDirectives::of(context, environment)?.directives,
            ]
            .concat(),
            openmp: OpenmpLinking::of(context, environment),
            include_debug_messages: context.include_debug_messages,
        })
    }

    /// Generates the bindings, compiles the wrappers, builds llama.cpp, and only then writes the
    /// cargo directives to `output`, with the OpenMP runtime `CMake` resolved linked after every
    /// planned library.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of the first step that fails.
    pub fn run(self, output: &mut impl Write) -> Result<(), BuildError> {
        let bindings_directives = self.bindings.generate()?;

        self.wrappers.compile()?;

        let cmake_cache = self.cmake.build()?;
        let openmp_directives = self.openmp.directives(&cmake_cache)?;

        emit_directives(
            &[self.directives, openmp_directives, bindings_directives].concat(),
            self.include_debug_messages,
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::ffi::OsStr;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process;

    use enumflags2::BitFlags;

    use llama_cpp_error_chain::error_chain::error_chain;

    use super::NativeBuild;
    use crate::bindings_generation::BindingsGeneration;
    use crate::build_context::tests::cargo_environment;
    use crate::build_environment::BuildEnvironment;
    use crate::build_environment::tests::variable;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_build::CmakeBuild;
    use crate::cmake_define::CmakeDefine;
    use crate::cmake_file_api::CmakeFileApi;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;
    use crate::openmp_linking::OpenmpLinking;
    use crate::openmp_runtime_source::OpenmpRuntimeSource;
    use crate::target_os::TargetOs;
    use crate::wrapper_compilation::WrapperCompilation;

    const HOST_TRIPLE: &str = env!("LLAMA_CPP_BINDINGS_BUILD_TARGET");

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
                cpp_standard: "c++17",
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
                file_api: CmakeFileApi {
                    build_dir: root.join("cmake/build"),
                },
                target_triple: HOST_TRIPLE.to_owned(),
                host_triple: HOST_TRIPLE.to_owned(),
                defines: vec![CmakeDefine::new("FIXTURE_LIBRARY_NAME", "fixture")],
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
            openmp: OpenmpLinking {
                source: OpenmpRuntimeSource::BundledGgml,
                target_os: TargetOs::Linux,
            },
            include_debug_messages: false,
        }
    }

    fn successful_directives(root: &Path) -> String {
        format!(
            "cargo:rustc-link-search=native={root}/cmake/lib\n\
             cargo:rustc-link-lib=static=fixture\n\
             cargo:rustc-link-search=native={root}/cmake/lib\n\
             cargo:rustc-link-lib=dylib:+verbatim=libfixture.a\n\
             cargo:rerun-if-env-changed=TARGET\n\
             cargo:rerun-if-env-changed=BINDGEN_EXTRA_CLANG_ARGS\n\
             cargo:rerun-if-changed={FIXTURES}/bindings/wrapper.h\n\
             cargo:rerun-if-changed={FIXTURES}/bindings/llama_types.h\n",
            root = root.display()
        )
    }

    fn build_root(name: &str) -> PathBuf {
        let root =
            env::temp_dir().join(format!("llama-cpp-bindings-build-{name}-{}", process::id()));

        fs::create_dir_all(&root).unwrap();

        root
    }

    const FIXTURE_CMAKE_PROJECT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cmake_project");

    fn failure(environment: &BuildEnvironment) -> [Option<String>; 2] {
        let mut output = Vec::new();

        [
            NativeBuild::from_environment(environment)
                .and_then(|native_build| native_build.run(&mut output))
                .err()
                .map(|error| error_chain(&error)),
            String::from_utf8(output).ok(),
        ]
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
                successful_directives(&root)
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
    fn a_build_tree_left_by_a_relocated_source_still_reports_the_cmake_cache() {
        let root = build_root("relocated-source");
        let relocated_source = root.join("relocated_cmake_project");
        let mut native_build = fixture_build(&root);
        let mut output = Vec::new();

        fixture_build(&root).run(&mut Vec::new()).unwrap();
        fs::create_dir_all(&relocated_source).unwrap();
        for file_name in ["CMakeLists.txt", "fixture.c"] {
            fs::copy(
                PathBuf::from(FIXTURES)
                    .join("cmake_project")
                    .join(file_name),
                relocated_source.join(file_name),
            )
            .unwrap();
        }
        native_build.cmake.source_dir = relocated_source;
        native_build.run(&mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            successful_directives(&root)
        );
        fs::remove_dir_all(root).unwrap();
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
                .map(|error| error_chain(&error)),
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
                .map(|error| error_chain(&error)),
            Some(format!(
                "filesystem operation failed for {MANIFEST}/backends: Not a directory (os error \
                 20)"
            ))
        );
        assert_eq!(output, Vec::<u8>::new());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_apple_build_whose_ggml_needs_an_openmp_runtime_emits_no_directive() {
        let root = build_root("apple-openmp");
        let mut native_build = fixture_build(&root);
        let mut output = Vec::new();

        native_build.openmp.target_os = TargetOs::Apple;

        assert_eq!(
            native_build
                .run(&mut output)
                .err()
                .map(|error| error_chain(&error)),
            Some(format!(
                "ggml needs the OpenMP runtime [{:?}], which Apple's linker cannot link by its \
                 exact file name",
                root.join("cmake/lib/libfixture.a")
            ))
        );
        assert_eq!(output, Vec::<u8>::new());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failure_at_any_stage_is_reported_and_nothing_is_emitted() {
        let mut without_target = cargo_environment(BitFlags::empty(), &[]);

        without_target.variables.remove(OsStr::new("TARGET"));

        assert_eq!(
            [
                failure(&without_target),
                failure(&cargo_environment(
                    BitFlags::from(BuildFeature::SystemGgml),
                    &[]
                )),
                failure(&cargo_environment(BitFlags::from(BuildFeature::Rocm), &[])),
                failure(&cargo_environment(
                    BitFlags::empty(),
                    &[variable("CARGO_MANIFEST_DIR", FIXTURE_CMAKE_PROJECT)]
                )),
            ],
            [
                [
                    Some(
                        "environment variable TARGET could not be read: environment variable not \
                         found"
                            .to_owned()
                    ),
                    Some(String::new())
                ],
                [
                    Some(
                        "environment variable GGML_DIR could not be read: environment variable \
                         not found"
                            .to_owned()
                    ),
                    Some(String::new())
                ],
                [
                    Some(
                        "environment variable ROCM_PATH could not be read: environment variable \
                         not found"
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
