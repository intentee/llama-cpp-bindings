use std::path::Path;
use std::path::PathBuf;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cargo_directive::CargoDirective;
use crate::link_kind::LinkKind;
use crate::native_library::NativeLibrary;
use crate::target_os::TargetOs;

const fn link(link_kind: LinkKind, name: &'static str) -> CargoDirective {
    CargoDirective::LinkLibrary(NativeLibrary { link_kind, name })
}

fn existing_directory(name: &'static str, path: PathBuf) -> Result<PathBuf, BuildError> {
    if path.is_dir() {
        Ok(path)
    } else {
        Err(BuildError::EnvironmentDirectory { name, path })
    }
}

fn cmake_search_paths(context: &BuildContext) -> Vec<CargoDirective> {
    let cmake_dir = context.cmake_dir();

    vec![
        CargoDirective::LinkSearch(cmake_dir.join("lib")),
        CargoDirective::LinkSearch(cmake_dir.join("lib64")),
        CargoDirective::LinkSearch(cmake_dir),
    ]
}

fn system_ggml_search_paths(
    environment: &BuildEnvironment,
) -> Result<Vec<CargoDirective>, BuildError> {
    if !environment.has(BuildFeature::SystemGgml) {
        return Ok(Vec::new());
    }

    let library_dir = existing_directory(
        "GGML_LIBRARY_DIR",
        PathBuf::from(environment.required("GGML_LIBRARY_DIR")?),
    )?;

    Ok(vec![
        CargoDirective::RerunIfEnvChanged("GGML_LIBRARY_DIR".to_owned()),
        CargoDirective::LinkSearchNative(library_dir),
    ])
}

fn private_dependency_search_paths(context: &BuildContext) -> Vec<CargoDirective> {
    let build_dir = context.cmake_dir().join("build");

    [
        build_dir.join("common"),
        build_dir.join("vendor").join("cpp-httplib"),
        build_dir.join("vendor").join("hash"),
    ]
    .into_iter()
    .flat_map(|library_dir| {
        [
            CargoDirective::LinkSearchNative(library_dir.clone()),
            CargoDirective::LinkSearchNative(library_dir.join(&context.profile)),
        ]
    })
    .collect()
}

fn native_libraries(context: &BuildContext, environment: &BuildEnvironment) -> Vec<NativeLibrary> {
    let cmake_kind = if context.build_shared_libs {
        LinkKind::Dylib
    } else {
        LinkKind::Static
    };
    let built = |name| NativeLibrary {
        link_kind: cmake_kind,
        name,
    };
    let bundled = |name| NativeLibrary {
        link_kind: LinkKind::Static,
        name,
    };
    let mut libraries = vec![built("llama-common")];

    if !context.build_shared_libs {
        libraries.extend([bundled("llama-common-base"), bundled("cpp-httplib")]);
    }

    libraries.push(built("mtmd"));

    if !context.build_shared_libs {
        libraries.push(bundled("vendor-hash"));
    }

    libraries.push(built("llama"));

    if environment.has(BuildFeature::SystemGgml) {
        let ggml_kind = if environment.has(BuildFeature::SystemGgmlStatic) {
            LinkKind::Static
        } else {
            LinkKind::Dylib
        };

        libraries.extend(["ggml-cpu", "ggml-base", "ggml"].map(|name| NativeLibrary {
            link_kind: ggml_kind,
            name,
        }));

        return libraries;
    }

    if !environment.has(BuildFeature::DynamicBackends) {
        if environment.has(BuildFeature::Cuda) {
            libraries.push(built("ggml-cuda"));
        }

        if environment.has(BuildFeature::Metal) {
            libraries.push(built("ggml-metal"));
        }

        if environment.has(BuildFeature::Vulkan) {
            libraries.push(built("ggml-vulkan"));
        }

        if environment.has(BuildFeature::Rocm) {
            libraries.push(built("ggml-hip"));
        }
    }

    libraries.extend(["ggml-cpu", "ggml-base", "ggml"].map(built));

    libraries
}

fn native_library_links(
    context: &BuildContext,
    environment: &BuildEnvironment,
) -> Vec<CargoDirective> {
    native_libraries(context, environment)
        .into_iter()
        .flat_map(|library| {
            let directive = CargoDirective::LinkLibrary(library);

            [
                CargoDirective::Debug(format!("LINK {directive}")),
                directive,
            ]
        })
        .collect()
}

fn cuda_links(context: &BuildContext, environment: &BuildEnvironment) -> Vec<CargoDirective> {
    if !environment.has(BuildFeature::Cuda) || context.build_shared_libs {
        return Vec::new();
    }

    let mut directives = vec![
        CargoDirective::RerunIfEnvChanged("CUDA_PATH".to_owned()),
        CargoDirective::RerunIfEnvChanged("CUDA_LIBRARY_PATH".to_owned()),
    ];

    directives.extend(
        environment
            .cuda_library_dirs
            .iter()
            .cloned()
            .map(CargoDirective::LinkSearchNative),
    );
    directives.extend(
        ["cudart_static", "cublas_static", "cublasLt_static"]
            .map(|name| link(LinkKind::Static, name)),
    );

    if !environment.has(BuildFeature::CudaNoVmm) {
        directives.push(link(LinkKind::Unspecified, "cuda"));
    }

    directives.push(link(LinkKind::Static, "culibos"));

    directives
}

fn rocm_links(
    context: &BuildContext,
    environment: &BuildEnvironment,
) -> Result<Vec<CargoDirective>, BuildError> {
    if !environment.has(BuildFeature::Rocm) || context.build_shared_libs {
        return Ok(Vec::new());
    }

    let rocm_lib = existing_directory(
        "ROCM_PATH",
        Path::new(&environment.required("ROCM_PATH")?).join("lib"),
    )?;

    Ok(vec![
        CargoDirective::RerunIfEnvChanged("ROCM_PATH".to_owned()),
        CargoDirective::LinkSearchNative(rocm_lib),
        link(LinkKind::Dylib, "amdhip64"),
        link(LinkKind::Dylib, "rocblas"),
        link(LinkKind::Dylib, "hipblas"),
    ])
}

fn openmp_links(context: &BuildContext, environment: &BuildEnvironment) -> Vec<CargoDirective> {
    if environment.has(BuildFeature::Openmp) && context.cargo_cfg_target_env == "gnu" {
        vec![link(LinkKind::Unspecified, "gomp")]
    } else {
        Vec::new()
    }
}

fn platform_links(context: &BuildContext, environment: &BuildEnvironment) -> Vec<CargoDirective> {
    match context.target_os {
        TargetOs::Linux => vec![link(LinkKind::Dylib, "stdc++")],
        TargetOs::Apple => vec![
            link(LinkKind::Framework, "Foundation"),
            link(LinkKind::Framework, "Metal"),
            link(LinkKind::Framework, "MetalKit"),
            link(LinkKind::Framework, "Accelerate"),
            link(LinkKind::Unspecified, "c++"),
        ],
        TargetOs::Android(_) if environment.has(BuildFeature::StaticStdcxx) => vec![
            link(LinkKind::Unspecified, "c++_static"),
            link(LinkKind::Unspecified, "c++abi"),
        ],
        TargetOs::Android(_) if environment.has(BuildFeature::SharedStdcxx) => {
            vec![link(LinkKind::Unspecified, "c++_shared")]
        }
        TargetOs::Android(_) => Vec::new(),
    }
}

/// The search paths and libraries that link the native build into the bindings, in the order
/// the linker resolves them.
///
/// # Errors
///
/// Returns [`BuildError::Environment`] when a library directory an enabled feature needs is not
/// set, or [`BuildError::EnvironmentDirectory`] when it does not name a directory.
pub fn link_directives(
    context: &BuildContext,
    environment: &BuildEnvironment,
) -> Result<Vec<CargoDirective>, BuildError> {
    Ok([
        cmake_search_paths(context),
        system_ggml_search_paths(environment)?,
        private_dependency_search_paths(context),
        native_library_links(context, environment),
        cuda_links(context, environment),
        rocm_links(context, environment)?,
        openmp_links(context, environment),
        platform_links(context, environment),
    ]
    .concat())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use super::link_directives;
    use crate::build_context::tests::ANDROID_OVERRIDES;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_environment::BuildEnvironment;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;

    const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures");

    fn linked(environment: &BuildEnvironment) -> Vec<CargoDirective> {
        link_directives(&context_of(environment), environment).unwrap()
    }

    fn libraries(directives: &[CargoDirective]) -> Vec<NativeLibrary> {
        directives
            .iter()
            .filter_map(|directive| match directive {
                CargoDirective::LinkLibrary(library) => Some(*library),
                _ => None,
            })
            .collect()
    }

    fn search_paths_and_rebuild_triggers(directives: &[CargoDirective]) -> Vec<CargoDirective> {
        directives
            .iter()
            .filter(|directive| {
                !directive.is_debug() && !matches!(directive, CargoDirective::LinkLibrary(_))
            })
            .cloned()
            .collect()
    }

    fn platform_libraries(
        features: BitFlags<BuildFeature>,
        overrides: &[(&str, &str)],
    ) -> Vec<NativeLibrary> {
        let libraries = libraries(&linked(&cargo_environment(features, overrides)));

        libraries
            .split(|library| library.name == "ggml")
            .next_back()
            .unwrap()
            .to_vec()
    }

    fn library(link_kind: LinkKind, name: &'static str) -> NativeLibrary {
        NativeLibrary { link_kind, name }
    }

    fn cmake_search_paths() -> Vec<CargoDirective> {
        vec![
            CargoDirective::LinkSearch(PathBuf::from("/out/cmake/lib")),
            CargoDirective::LinkSearch(PathBuf::from("/out/cmake/lib64")),
            CargoDirective::LinkSearch(PathBuf::from("/out/cmake")),
        ]
    }

    fn private_dependency_search_paths() -> Vec<CargoDirective> {
        [
            "/out/cmake/build/common",
            "/out/cmake/build/common/Release",
            "/out/cmake/build/vendor/cpp-httplib",
            "/out/cmake/build/vendor/cpp-httplib/Release",
            "/out/cmake/build/vendor/hash",
            "/out/cmake/build/vendor/hash/Release",
        ]
        .map(|path| CargoDirective::LinkSearchNative(PathBuf::from(path)))
        .to_vec()
    }

    fn static_llama_libraries() -> Vec<NativeLibrary> {
        [
            "llama-common",
            "llama-common-base",
            "cpp-httplib",
            "mtmd",
            "vendor-hash",
            "llama",
        ]
        .map(|name| library(LinkKind::Static, name))
        .to_vec()
    }

    #[test]
    fn a_static_linux_build_links_every_archive_after_its_dependents_and_then_libstdcxx() {
        let static_archives = [
            static_llama_libraries(),
            ["ggml-cpu", "ggml-base", "ggml"]
                .map(|name| library(LinkKind::Static, name))
                .to_vec(),
        ]
        .concat()
        .into_iter()
        .flat_map(|archive| {
            let directive = CargoDirective::LinkLibrary(archive);

            [
                CargoDirective::Debug(format!("LINK {directive}")),
                directive,
            ]
        })
        .collect::<Vec<CargoDirective>>();

        assert_eq!(
            linked(&cargo_environment(BitFlags::empty(), &[])),
            [
                cmake_search_paths(),
                private_dependency_search_paths(),
                static_archives,
                vec![CargoDirective::LinkLibrary(library(
                    LinkKind::Dylib,
                    "stdc++"
                ))],
            ]
            .concat()
        );
    }

    #[test]
    fn a_dynamic_build_links_shared_libraries_and_leaves_gpu_backends_to_runtime_loading() {
        assert_eq!(
            libraries(&linked(&cargo_environment(
                BuildFeature::DynamicLink
                    | BuildFeature::DynamicBackends
                    | BuildFeature::Cuda
                    | BuildFeature::Rocm,
                &[]
            ))),
            [
                "llama-common",
                "mtmd",
                "llama",
                "ggml-cpu",
                "ggml-base",
                "ggml",
                "stdc++"
            ]
            .map(|name| library(LinkKind::Dylib, name))
        );
    }

    #[test]
    fn static_gpu_backends_link_their_archives_and_the_vendor_runtimes() {
        let rocm_root = std::env::temp_dir().join(format!(
            "llama-cpp-bindings-build-rocm-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(rocm_root.join("lib")).unwrap();
        let environment = BuildEnvironment {
            cuda_library_dirs: vec![PathBuf::from("/cuda/lib64")],
            ..cargo_environment(
                BuildFeature::Cuda
                    | BuildFeature::Metal
                    | BuildFeature::Vulkan
                    | BuildFeature::Rocm,
                &[("ROCM_PATH", rocm_root.to_str().unwrap())],
            )
        };
        let directives = linked(&environment);

        assert_eq!(
            libraries(&directives),
            [
                static_llama_libraries(),
                [
                    "ggml-cuda",
                    "ggml-metal",
                    "ggml-vulkan",
                    "ggml-hip",
                    "ggml-cpu",
                    "ggml-base",
                    "ggml",
                    "cudart_static",
                    "cublas_static",
                    "cublasLt_static",
                ]
                .map(|name| library(LinkKind::Static, name))
                .to_vec(),
                vec![
                    library(LinkKind::Unspecified, "cuda"),
                    library(LinkKind::Static, "culibos"),
                    library(LinkKind::Dylib, "amdhip64"),
                    library(LinkKind::Dylib, "rocblas"),
                    library(LinkKind::Dylib, "hipblas"),
                    library(LinkKind::Dylib, "stdc++"),
                ],
            ]
            .concat()
        );
        assert_eq!(
            search_paths_and_rebuild_triggers(&directives),
            [
                cmake_search_paths(),
                private_dependency_search_paths(),
                vec![
                    CargoDirective::RerunIfEnvChanged("CUDA_PATH".to_owned()),
                    CargoDirective::RerunIfEnvChanged("CUDA_LIBRARY_PATH".to_owned()),
                    CargoDirective::LinkSearchNative(PathBuf::from("/cuda/lib64")),
                    CargoDirective::RerunIfEnvChanged("ROCM_PATH".to_owned()),
                    CargoDirective::LinkSearchNative(rocm_root.join("lib")),
                ],
            ]
            .concat()
        );
        std::fs::remove_dir_all(rocm_root).unwrap();
    }

    #[test]
    fn cuda_without_virtual_memory_management_does_not_link_the_driver() {
        assert_eq!(
            libraries(&linked(&cargo_environment(
                BuildFeature::Cuda | BuildFeature::CudaNoVmm,
                &[]
            ))),
            [
                static_llama_libraries(),
                [
                    "ggml-cuda",
                    "ggml-cpu",
                    "ggml-base",
                    "ggml",
                    "cudart_static",
                    "cublas_static",
                    "cublasLt_static",
                    "culibos",
                ]
                .map(|name| library(LinkKind::Static, name))
                .to_vec(),
                vec![library(LinkKind::Dylib, "stdc++")],
            ]
            .concat()
        );
    }

    #[test]
    fn system_ggml_links_the_installed_ggml_instead_of_any_built_backend() {
        let system_ggml_linked = |features: BitFlags<BuildFeature>| {
            linked(&cargo_environment(
                features | BuildFeature::SystemGgml | BuildFeature::Cuda,
                &[("GGML_LIBRARY_DIR", FIXTURES)],
            ))
        };
        let linked_with_ggml_as = |ggml_kind: LinkKind| {
            [
                static_llama_libraries(),
                ["ggml-cpu", "ggml-base", "ggml"]
                    .map(|name| library(ggml_kind, name))
                    .to_vec(),
                ["cudart_static", "cublas_static", "cublasLt_static"]
                    .map(|name| library(LinkKind::Static, name))
                    .to_vec(),
                vec![
                    library(LinkKind::Unspecified, "cuda"),
                    library(LinkKind::Static, "culibos"),
                    library(LinkKind::Dylib, "stdc++"),
                ],
            ]
            .concat()
        };
        let shared_ggml = system_ggml_linked(BitFlags::empty());

        assert_eq!(
            [
                libraries(&shared_ggml),
                libraries(&system_ggml_linked(BitFlags::from(
                    BuildFeature::SystemGgmlStatic
                ))),
            ],
            [
                linked_with_ggml_as(LinkKind::Dylib),
                linked_with_ggml_as(LinkKind::Static)
            ]
        );
        assert_eq!(
            search_paths_and_rebuild_triggers(&shared_ggml),
            [
                cmake_search_paths(),
                vec![
                    CargoDirective::RerunIfEnvChanged("GGML_LIBRARY_DIR".to_owned()),
                    CargoDirective::LinkSearchNative(PathBuf::from(FIXTURES)),
                ],
                private_dependency_search_paths(),
                vec![
                    CargoDirective::RerunIfEnvChanged("CUDA_PATH".to_owned()),
                    CargoDirective::RerunIfEnvChanged("CUDA_LIBRARY_PATH".to_owned()),
                ],
            ]
            .concat()
        );
    }

    #[test]
    fn openmp_links_libgomp_only_for_gnu_targets() {
        assert_eq!(
            ["gnu", "musl"].map(|cargo_cfg_target_env| {
                platform_libraries(
                    BitFlags::from(BuildFeature::Openmp),
                    &[("CARGO_CFG_TARGET_ENV", cargo_cfg_target_env)],
                )
            }),
            [
                vec![
                    library(LinkKind::Unspecified, "gomp"),
                    library(LinkKind::Dylib, "stdc++"),
                ],
                vec![library(LinkKind::Dylib, "stdc++")],
            ]
        );
    }

    #[test]
    fn apple_and_android_targets_link_their_platform_cxx_runtime() {
        assert_eq!(
            [
                platform_libraries(BitFlags::empty(), &[("CARGO_CFG_TARGET_OS", "macos")]),
                platform_libraries(
                    BitFlags::from(BuildFeature::StaticStdcxx),
                    &ANDROID_OVERRIDES
                ),
                platform_libraries(
                    BitFlags::from(BuildFeature::SharedStdcxx),
                    &ANDROID_OVERRIDES
                ),
                platform_libraries(BitFlags::empty(), &ANDROID_OVERRIDES),
            ],
            [
                vec![
                    library(LinkKind::Framework, "Foundation"),
                    library(LinkKind::Framework, "Metal"),
                    library(LinkKind::Framework, "MetalKit"),
                    library(LinkKind::Framework, "Accelerate"),
                    library(LinkKind::Unspecified, "c++"),
                ],
                vec![
                    library(LinkKind::Unspecified, "c++_static"),
                    library(LinkKind::Unspecified, "c++abi"),
                ],
                vec![library(LinkKind::Unspecified, "c++_shared")],
                Vec::new(),
            ]
        );
    }

    #[test]
    fn library_directories_an_enabled_feature_needs_must_exist() {
        let missing_directory = std::env::temp_dir().join(format!(
            "llama-cpp-bindings-build-missing-{}",
            std::process::id()
        ));
        let refusal = |features: BitFlags<BuildFeature>, overrides: &[(&str, &str)]| {
            let environment = cargo_environment(features, overrides);

            link_directives(&context_of(&environment), &environment)
                .err()
                .map(|error| error.to_string())
        };

        assert_eq!(
            [
                refusal(BitFlags::from(BuildFeature::SystemGgml), &[]),
                refusal(
                    BitFlags::from(BuildFeature::SystemGgml),
                    &[("GGML_LIBRARY_DIR", missing_directory.to_str().unwrap())]
                ),
                refusal(BitFlags::from(BuildFeature::Rocm), &[]),
                refusal(
                    BitFlags::from(BuildFeature::Rocm),
                    &[("ROCM_PATH", env!("CARGO_MANIFEST_DIR"))]
                ),
            ],
            [
                Some(
                    "environment variable GGML_LIBRARY_DIR could not be read: environment \
                     variable not found"
                        .to_owned()
                ),
                Some(format!(
                    "environment path GGML_LIBRARY_DIR does not name a directory: {}",
                    missing_directory.display()
                )),
                Some(
                    "environment variable ROCM_PATH could not be read: environment variable not \
                     found"
                        .to_owned()
                ),
                Some(format!(
                    "environment path ROCM_PATH does not name a directory: {}",
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("lib").display()
                )),
            ]
        );
    }
}
