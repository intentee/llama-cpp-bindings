use std::io::Write;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::cmake_build::CmakeBuild;
use crate::cmake_configuration::CmakeConfiguration;
use crate::cmake_file_api::CmakeFileApi;
use crate::constants::LLAMA_CPP_DIR;
use crate::constants::LLAMA_CPP_ROOT_METADATA_KEY;
use crate::emit_directives::emit_directives;
use crate::link_directives::LinkDirectives;
use crate::openmp_linking::OpenmpLinking;
use crate::planned_build::PlannedBuild;

/// Every step of building the llama.cpp sources of the native package with `CMake`, decided before
/// any of them runs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeLibraryBuild {
    pub cmake: CmakeBuild,
    pub directives: Vec<CargoDirective>,
    pub openmp: OpenmpLinking,
    pub include_debug_messages: bool,
}

impl PlannedBuild for NativeLibraryBuild {
    /// Plans the `CMake` build of the llama.cpp sources under the package, and hands their
    /// directory to the build scripts of the packages that depend on it.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`CmakeConfiguration::of`] or [`LinkDirectives::of`].
    fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let CmakeConfiguration {
            defines,
            required_directories,
            directives: cmake_directives,
        } = CmakeConfiguration::of(context, environment)?;
        let llama_cpp_dir = context.manifest_dir.join(LLAMA_CPP_DIR);

        Ok(Self {
            cmake: CmakeBuild {
                source_dir: llama_cpp_dir.clone(),
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
                vec![
                    context.build_script_trigger(),
                    CargoDirective::RerunIfEnvChanged("CUDAARCHS".to_owned()),
                    CargoDirective::RerunIfChanged(llama_cpp_dir.clone()),
                ],
                cmake_directives,
                LinkDirectives::of(context, environment)?.directives,
                vec![CargoDirective::LinksMetadata {
                    key: LLAMA_CPP_ROOT_METADATA_KEY,
                    value: llama_cpp_dir,
                }],
            ]
            .concat(),
            openmp: OpenmpLinking::of(context, environment),
            include_debug_messages: context.include_debug_messages,
        })
    }

    /// Builds llama.cpp, and only then writes the cargo directives to `output`, with the OpenMP
    /// runtime `CMake` resolved linked after every planned library.
    ///
    /// # Errors
    ///
    /// Returns the [`BuildError`] of the first step that fails.
    fn run(self, output: &mut impl Write) -> Result<(), BuildError> {
        let cmake_cache = self.cmake.build()?;
        let openmp_directives = self.openmp.directives(&cmake_cache)?;

        emit_directives(
            &[self.directives, openmp_directives].concat(),
            self.include_debug_messages,
            output,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use super::NativeLibraryBuild;
    use crate::build_context::BuildContext;
    use crate::build_context::tests::FIXTURES;
    use crate::build_context::tests::HOST_TRIPLE;
    use crate::build_context::tests::MANIFEST;
    use crate::build_context::tests::build_root;
    use crate::build_context::tests::cargo_environment;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_build::CmakeBuild;
    use crate::cmake_define::CmakeDefine;
    use crate::cmake_file_api::CmakeFileApi;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;
    use crate::openmp_linking::OpenmpLinking;
    use crate::openmp_runtime_source::OpenmpRuntimeSource;
    use crate::planned_build::PlannedBuild;
    use crate::planned_build::tests::run_failure;
    use crate::target_os::TargetOs;

    fn fixture_build(root: &Path) -> NativeLibraryBuild {
        NativeLibraryBuild {
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
             cargo:rustc-link-lib=dylib:+verbatim=libfixture.a\n",
            root = root.display()
        )
    }

    #[test]
    fn llama_cpp_is_built_from_the_package_and_its_directory_is_handed_to_dependents() {
        let environment = cargo_environment(BitFlags::empty(), &[]);
        let native_library_build =
            NativeLibraryBuild::of(&BuildContext::of_linux_host(&environment), &environment)
                .unwrap();

        assert_eq!(
            native_library_build.cmake.source_dir,
            PathBuf::from("/sys/llama.cpp")
        );
        assert_eq!(
            native_library_build
                .directives
                .into_iter()
                .filter(|directive| matches!(
                    directive,
                    CargoDirective::RerunIfChanged(_)
                        | CargoDirective::RerunIfEnvChanged(_)
                        | CargoDirective::LinksMetadata { .. }
                ))
                .collect::<Vec<CargoDirective>>(),
            [
                CargoDirective::RerunIfChanged(PathBuf::from("/sys/build.rs")),
                CargoDirective::RerunIfEnvChanged("CUDAARCHS".to_owned()),
                CargoDirective::RerunIfChanged(PathBuf::from("/sys/llama.cpp")),
                CargoDirective::LinksMetadata {
                    key: "root",
                    value: PathBuf::from("/sys/llama.cpp"),
                },
            ]
        );
    }

    #[test]
    fn a_successful_build_writes_the_libraries_then_the_directives() {
        let root = build_root("successful");
        let mut output = Vec::new();

        fixture_build(&root).run(&mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            successful_directives(&root)
        );
        assert_eq!(
            [
                root.join("cmake/lib/libfixture.a").is_file(),
                root.join("cmake/backends").is_dir(),
            ],
            [true; 2]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_build_tree_left_by_a_relocated_source_still_reports_the_cmake_cache() {
        let root = build_root("relocated-source");
        let relocated_source = root.join("relocated_cmake_project");
        let mut native_library_build = fixture_build(&root);
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
        native_library_build.cmake.source_dir = relocated_source;
        native_library_build.run(&mut output).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            successful_directives(&root)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_cmake_build_that_cannot_create_its_directories_emits_no_directive() {
        let root = build_root("failed-cmake");
        let mut native_library_build = fixture_build(&root);

        native_library_build.cmake.required_directories =
            vec![PathBuf::from(MANIFEST).join("backends")];

        assert_eq!(
            run_failure(native_library_build),
            [
                Some(format!(
                    "filesystem operation failed for {MANIFEST}/backends: Not a directory (os error \
                     20)"
                )),
                Some(String::new())
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn an_apple_build_whose_ggml_needs_an_openmp_runtime_emits_no_directive() {
        let root = build_root("apple-openmp");
        let mut native_library_build = fixture_build(&root);

        native_library_build.openmp.target_os = TargetOs::Apple;

        assert_eq!(
            run_failure(native_library_build),
            [
                Some(format!(
                    "ggml needs the OpenMP runtime [{:?}], which Apple's linker cannot link by its \
                     exact file name",
                    root.join("cmake/lib/libfixture.a")
                )),
                Some(String::new())
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_failure_at_any_planning_stage_is_reported_and_nothing_is_emitted() {
        let mut without_target = cargo_environment(BitFlags::empty(), &[]);

        without_target.variables.remove(OsStr::new("TARGET"));

        assert_eq!(
            [
                without_target.build_failure::<NativeLibraryBuild>(),
                cargo_environment(BitFlags::from(BuildFeature::SystemGgml), &[])
                    .build_failure::<NativeLibraryBuild>(),
                cargo_environment(BitFlags::from(BuildFeature::Rocm), &[])
                    .build_failure::<NativeLibraryBuild>(),
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
            ]
        );
    }
}
