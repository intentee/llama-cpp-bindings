use std::path::PathBuf;

use crate::android_ndk::AndroidNdk;
use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cargo_directive::CargoDirective;
use crate::cmake_define::CmakeDefine;
use crate::cpu_tuning::CpuTuning;
use crate::link_kind::LinkKind;
use crate::native_library::NativeLibrary;
use crate::target_os::TargetOs;

fn base_defines() -> Vec<CmakeDefine> {
    vec![
        CmakeDefine::new("LLAMA_BUILD_TESTS", "OFF"),
        CmakeDefine::new("LLAMA_BUILD_EXAMPLES", "OFF"),
        CmakeDefine::new("LLAMA_BUILD_SERVER", "OFF"),
        CmakeDefine::new("LLAMA_BUILD_TOOLS", "OFF"),
        CmakeDefine::new("LLAMA_BUILD_APP", "OFF"),
        CmakeDefine::new("LLAMA_BUILD_COMMON", "ON"),
        CmakeDefine::new("LLAMA_BUILD_MTMD", "ON"),
        CmakeDefine::new("MTMD_VIDEO", "OFF"),
        CmakeDefine::new("LLAMA_OPENSSL", "OFF"),
    ]
}

/// How llama.cpp is configured with `CMake` for this build, and the cargo directives that
/// configuration implies.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CmakeConfiguration {
    pub defines: Vec<CmakeDefine>,
    pub compiler_flags: Vec<String>,
    pub required_directories: Vec<PathBuf>,
    pub directives: Vec<CargoDirective>,
}

impl CmakeConfiguration {
    /// # Errors
    ///
    /// Returns the [`BuildError`] of [`CpuTuning::of`], or [`BuildError::Environment`] when a
    /// variable an enabled feature reads is not Unicode, or when `GGML_DIR` is missing for
    /// `system-ggml`.
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let CpuTuning {
            defines: cpu_defines,
            compiler_flags,
            debug_messages,
        } = CpuTuning::of(context, environment)?;
        let mut configuration = Self {
            defines: [base_defines(), cpu_defines].concat(),
            compiler_flags,
            required_directories: Vec::new(),
            directives: debug_messages,
        };

        configuration.defines.push(CmakeDefine::switch(
            "BUILD_SHARED_LIBS",
            context.build_shared_libs,
        ));
        configuration.configure_platform(context, environment);
        configuration.configure_gpu_backends(context, environment)?;
        configuration.defines.push(CmakeDefine::switch(
            "GGML_OPENMP",
            environment.has(BuildFeature::Openmp)
                && !matches!(context.target_os, TargetOs::Android(_)),
        ));
        configuration.configure_system_ggml(environment)?;
        configuration.configure_dynamic_backends(context, environment);

        Ok(configuration)
    }

    fn configure_platform(&mut self, context: &BuildContext, environment: &BuildEnvironment) {
        match &context.target_os {
            TargetOs::Apple => self.defines.push(CmakeDefine::new("GGML_BLAS", "OFF")),
            TargetOs::Android(android_ndk) => self.configure_android(android_ndk, environment),
            TargetOs::Linux => {}
        }
    }

    fn configure_android(&mut self, android_ndk: &AndroidNdk, environment: &BuildEnvironment) {
        self.directives.extend([
            CargoDirective::RerunIfEnvChanged("ANDROID_NDK_HOME".to_owned()),
            CargoDirective::RerunIfEnvChanged("ANDROID_PLATFORM".to_owned()),
        ]);
        self.defines.extend([
            CmakeDefine::new("CMAKE_TOOLCHAIN_FILE", android_ndk.cmake_toolchain_file()),
            CmakeDefine::new("ANDROID_PLATFORM", android_ndk.android_platform()),
            CmakeDefine::new("ANDROID_ABI", android_ndk.abi),
        ]);

        if environment.has(BuildFeature::StaticStdcxx) {
            self.defines
                .push(CmakeDefine::new("ANDROID_STL", "c++_static"));
        } else if environment.has(BuildFeature::SharedStdcxx) {
            self.defines
                .push(CmakeDefine::new("ANDROID_STL", "c++_shared"));
        }

        self.defines.push(CmakeDefine::new("GGML_LLAMAFILE", "OFF"));
        self.directives.extend(["log", "android"].map(|name| {
            CargoDirective::LinkLibrary(NativeLibrary {
                link_kind: LinkKind::Unspecified,
                name,
            })
        }));
    }

    fn configure_gpu_backends(
        &mut self,
        context: &BuildContext,
        environment: &BuildEnvironment,
    ) -> Result<(), BuildError> {
        if environment.has(BuildFeature::Vulkan) {
            self.defines.push(CmakeDefine::new("GGML_VULKAN", "ON"));

            if context.target_os == TargetOs::Linux {
                self.link_vulkan(environment)?;
            }
        }

        if environment.has(BuildFeature::Cuda) {
            self.defines.push(CmakeDefine::new("GGML_CUDA", "ON"));

            if environment.has(BuildFeature::CudaNoVmm) {
                self.defines
                    .push(CmakeDefine::new("GGML_CUDA_NO_VMM", "ON"));
            }

            if let Some(cuda_architectures) = environment.optional("CUDAARCHS")? {
                self.defines.push(CmakeDefine::new(
                    "CMAKE_CUDA_ARCHITECTURES",
                    cuda_architectures,
                ));
            }
        }

        if environment.has(BuildFeature::Rocm) {
            self.defines.push(CmakeDefine::new("GGML_HIP", "ON"));
        }

        Ok(())
    }

    fn link_vulkan(&mut self, environment: &BuildEnvironment) -> Result<(), BuildError> {
        self.directives
            .push(CargoDirective::RerunIfEnvChanged("VULKAN_SDK".to_owned()));

        if let Some(vulkan_sdk) = environment.optional("VULKAN_SDK")? {
            self.directives.push(CargoDirective::LinkSearch(
                PathBuf::from(vulkan_sdk).join("lib"),
            ));
        }

        self.directives
            .push(CargoDirective::LinkLibrary(NativeLibrary {
                link_kind: LinkKind::Unspecified,
                name: "vulkan",
            }));

        Ok(())
    }

    fn configure_system_ggml(&mut self, environment: &BuildEnvironment) -> Result<(), BuildError> {
        if environment.has(BuildFeature::SystemGgml) {
            self.directives
                .push(CargoDirective::RerunIfEnvChanged("GGML_DIR".to_owned()));

            let ggml_dir = environment.required("GGML_DIR")?;

            self.defines.extend([
                CmakeDefine::new("LLAMA_USE_SYSTEM_GGML", "ON"),
                CmakeDefine::new("GGML_DIR", ggml_dir),
            ]);
        }

        Ok(())
    }

    fn configure_dynamic_backends(
        &mut self,
        context: &BuildContext,
        environment: &BuildEnvironment,
    ) {
        if environment.has(BuildFeature::DynamicBackends) {
            let backends_dir = context.cmake_dir().join("backends");

            self.defines.extend([
                CmakeDefine::new("GGML_BACKEND_DL", "ON"),
                CmakeDefine::new("GGML_CPU_ALL_VARIANTS", "ON"),
                CmakeDefine::new("GGML_BACKEND_DIR", backends_dir.clone()),
            ]);
            self.directives.push(CargoDirective::Metadata {
                key: "backends_dir",
                value: backends_dir.display().to_string(),
            });
            self.required_directories.push(backends_dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use super::CmakeConfiguration;
    use crate::android_ndk::tests::FIXTURE_ANDROID_NDK;
    use crate::build_context::tests::ANDROID_OVERRIDES;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_environment::tests::with_non_unicode;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_define::CmakeDefine;
    use crate::link_kind::LinkKind;
    use crate::native_library::NativeLibrary;

    fn configured(
        features: BitFlags<BuildFeature>,
        overrides: &[(&str, &str)],
    ) -> CmakeConfiguration {
        let environment = cargo_environment(features, overrides);

        CmakeConfiguration::of(&context_of(&environment), &environment).unwrap()
    }

    fn after_the_base_switches(
        features: BitFlags<BuildFeature>,
        overrides: &[(&str, &str)],
    ) -> CmakeConfiguration {
        let mut configuration = configured(features, overrides);

        configuration.defines.drain(..9);

        configuration
    }

    fn compiling_with_no_target_features() -> CargoDirective {
        CargoDirective::Debug("Compiling with target features: ".to_owned())
    }

    fn unspecified_link(name: &'static str) -> CargoDirective {
        CargoDirective::LinkLibrary(NativeLibrary {
            link_kind: LinkKind::Unspecified,
            name,
        })
    }

    #[test]
    fn a_plain_linux_build_turns_off_everything_but_the_libraries_the_bindings_use() {
        assert_eq!(
            configured(BitFlags::empty(), &[]),
            CmakeConfiguration {
                defines: vec![
                    CmakeDefine::new("LLAMA_BUILD_TESTS", "OFF"),
                    CmakeDefine::new("LLAMA_BUILD_EXAMPLES", "OFF"),
                    CmakeDefine::new("LLAMA_BUILD_SERVER", "OFF"),
                    CmakeDefine::new("LLAMA_BUILD_TOOLS", "OFF"),
                    CmakeDefine::new("LLAMA_BUILD_APP", "OFF"),
                    CmakeDefine::new("LLAMA_BUILD_COMMON", "ON"),
                    CmakeDefine::new("LLAMA_BUILD_MTMD", "ON"),
                    CmakeDefine::new("MTMD_VIDEO", "OFF"),
                    CmakeDefine::new("LLAMA_OPENSSL", "OFF"),
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                ],
                compiler_flags: Vec::new(),
                required_directories: Vec::new(),
                directives: vec![compiling_with_no_target_features()],
            }
        );
    }

    #[test]
    fn apple_targets_build_without_blas_and_leave_vulkan_linking_to_the_platform() {
        assert_eq!(
            after_the_base_switches(
                BuildFeature::Metal | BuildFeature::Vulkan,
                &[("CARGO_CFG_TARGET_OS", "macos")]
            ),
            CmakeConfiguration {
                defines: vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                    CmakeDefine::new("GGML_BLAS", "OFF"),
                    CmakeDefine::new("GGML_VULKAN", "ON"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                ],
                compiler_flags: Vec::new(),
                required_directories: Vec::new(),
                directives: vec![compiling_with_no_target_features()],
            }
        );
    }

    #[test]
    fn android_targets_build_with_the_ndk_toolchain_their_stl_and_no_openmp() {
        let android_with = |features: BitFlags<BuildFeature>| {
            after_the_base_switches(features | BuildFeature::Openmp, &ANDROID_OVERRIDES)
        };
        let android_defines = |stl: Vec<CmakeDefine>| {
            [
                vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                    CmakeDefine::new(
                        "CMAKE_TOOLCHAIN_FILE",
                        format!("{FIXTURE_ANDROID_NDK}/build/cmake/android.toolchain.cmake"),
                    ),
                    CmakeDefine::new("ANDROID_PLATFORM", "android-28"),
                    CmakeDefine::new("ANDROID_ABI", "arm64-v8a"),
                ],
                stl,
                vec![
                    CmakeDefine::new("GGML_LLAMAFILE", "OFF"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                ],
            ]
            .concat()
        };
        let android_configuration = |stl: Vec<CmakeDefine>| CmakeConfiguration {
            defines: android_defines(stl),
            compiler_flags: Vec::new(),
            required_directories: Vec::new(),
            directives: vec![
                compiling_with_no_target_features(),
                CargoDirective::RerunIfEnvChanged("ANDROID_NDK_HOME".to_owned()),
                CargoDirective::RerunIfEnvChanged("ANDROID_PLATFORM".to_owned()),
                unspecified_link("log"),
                unspecified_link("android"),
            ],
        };

        assert_eq!(
            [
                android_with(BitFlags::from(BuildFeature::StaticStdcxx)),
                android_with(BitFlags::from(BuildFeature::SharedStdcxx)),
                android_with(BitFlags::empty()),
            ],
            [
                android_configuration(vec![CmakeDefine::new("ANDROID_STL", "c++_static")]),
                android_configuration(vec![CmakeDefine::new("ANDROID_STL", "c++_shared")]),
                android_configuration(Vec::new()),
            ]
        );
    }

    #[test]
    fn gpu_backends_turn_on_their_ggml_switches_and_take_their_sdk_and_architectures_when_set() {
        let gpu_configuration = |overrides: &[(&str, &str)]| {
            after_the_base_switches(
                BuildFeature::Vulkan
                    | BuildFeature::Cuda
                    | BuildFeature::CudaNoVmm
                    | BuildFeature::Rocm,
                overrides,
            )
        };
        let gpu_defines = |cuda_architectures: Vec<CmakeDefine>| {
            [
                vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                    CmakeDefine::new("GGML_VULKAN", "ON"),
                    CmakeDefine::new("GGML_CUDA", "ON"),
                    CmakeDefine::new("GGML_CUDA_NO_VMM", "ON"),
                ],
                cuda_architectures,
                vec![
                    CmakeDefine::new("GGML_HIP", "ON"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                ],
            ]
            .concat()
        };
        let vulkan_directives = |vulkan_sdk_search_path: Vec<CargoDirective>| {
            [
                vec![
                    compiling_with_no_target_features(),
                    CargoDirective::RerunIfEnvChanged("VULKAN_SDK".to_owned()),
                ],
                vulkan_sdk_search_path,
                vec![unspecified_link("vulkan")],
            ]
            .concat()
        };

        assert_eq!(
            [
                gpu_configuration(&[("VULKAN_SDK", "/vulkan"), ("CUDAARCHS", "89")]),
                gpu_configuration(&[]),
            ],
            [
                CmakeConfiguration {
                    defines: gpu_defines(vec![CmakeDefine::new("CMAKE_CUDA_ARCHITECTURES", "89")]),
                    compiler_flags: Vec::new(),
                    required_directories: Vec::new(),
                    directives: vulkan_directives(vec![CargoDirective::LinkSearch(PathBuf::from(
                        "/vulkan/lib"
                    ))]),
                },
                CmakeConfiguration {
                    defines: gpu_defines(Vec::new()),
                    compiler_flags: Vec::new(),
                    required_directories: Vec::new(),
                    directives: vulkan_directives(Vec::new()),
                },
            ]
        );
    }

    #[test]
    fn openmp_is_turned_on_for_linux() {
        assert_eq!(
            after_the_base_switches(BitFlags::from(BuildFeature::Openmp), &[]).defines,
            [
                CmakeDefine::new("GGML_NATIVE", "OFF"),
                CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                CmakeDefine::new("GGML_OPENMP", "ON"),
            ]
        );
    }

    #[test]
    fn system_ggml_builds_llama_cpp_against_the_ggml_in_ggml_dir() {
        assert_eq!(
            after_the_base_switches(
                BitFlags::from(BuildFeature::SystemGgml),
                &[("GGML_DIR", "/ggml/lib/cmake/ggml")]
            ),
            CmakeConfiguration {
                defines: vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "OFF"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                    CmakeDefine::new("LLAMA_USE_SYSTEM_GGML", "ON"),
                    CmakeDefine::new("GGML_DIR", "/ggml/lib/cmake/ggml"),
                ],
                compiler_flags: Vec::new(),
                required_directories: Vec::new(),
                directives: vec![
                    compiling_with_no_target_features(),
                    CargoDirective::RerunIfEnvChanged("GGML_DIR".to_owned()),
                ],
            }
        );
    }

    #[test]
    fn dynamic_backends_are_built_as_shared_libraries_installed_into_a_published_backends_dir() {
        assert_eq!(
            after_the_base_switches(
                BuildFeature::DynamicBackends | BuildFeature::DynamicLink,
                &[]
            ),
            CmakeConfiguration {
                defines: vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("BUILD_SHARED_LIBS", "ON"),
                    CmakeDefine::new("GGML_OPENMP", "OFF"),
                    CmakeDefine::new("GGML_BACKEND_DL", "ON"),
                    CmakeDefine::new("GGML_CPU_ALL_VARIANTS", "ON"),
                    CmakeDefine::new("GGML_BACKEND_DIR", "/out/cmake/backends"),
                ],
                compiler_flags: Vec::new(),
                required_directories: vec![PathBuf::from("/out/cmake/backends")],
                directives: vec![
                    compiling_with_no_target_features(),
                    CargoDirective::Metadata {
                        key: "backends_dir",
                        value: "/out/cmake/backends".to_owned(),
                    },
                ],
            }
        );
    }

    #[test]
    fn variables_the_configuration_reads_must_be_present_and_unicode() {
        let context = context_of(&cargo_environment(BitFlags::empty(), &[]));
        let refusal = |features: BitFlags<BuildFeature>, name: &str| {
            CmakeConfiguration::of(
                &context,
                &with_non_unicode(cargo_environment(features, &[]), name),
            )
            .err()
            .map(|error| error.to_string())
        };
        let not_unicode = |name: &str| {
            Some(format!(
                "environment variable {name} could not be read: environment variable was not \
                 valid unicode: \"\\xFF\""
            ))
        };

        assert_eq!(
            [
                refusal(BitFlags::empty(), "CARGO_ENCODED_RUSTFLAGS"),
                refusal(BitFlags::from(BuildFeature::Vulkan), "VULKAN_SDK"),
                refusal(BitFlags::from(BuildFeature::Cuda), "CUDAARCHS"),
                CmakeConfiguration::of(
                    &context,
                    &cargo_environment(BitFlags::from(BuildFeature::SystemGgml), &[])
                )
                .err()
                .map(|error| error.to_string()),
            ],
            [
                not_unicode("CARGO_ENCODED_RUSTFLAGS"),
                not_unicode("VULKAN_SDK"),
                not_unicode("CUDAARCHS"),
                Some(
                    "environment variable GGML_DIR could not be read: environment variable not \
                     found"
                        .to_owned()
                ),
            ]
        );
    }
}
