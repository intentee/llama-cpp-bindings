use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::cargo_directive::CargoDirective;
use crate::cmake_define::CmakeDefine;
use crate::target_os::TargetOs;

fn codegen_options(encoded_rustflags: &str) -> Vec<&str> {
    let mut options = Vec::new();
    let mut flags = encoded_rustflags.split('\x1f');

    while let Some(flag) = flags.next() {
        if flag == "-C" || flag == "--codegen" {
            options.extend(flags.next());
        } else if let Some(option) = flag
            .strip_prefix("--codegen=")
            .or_else(|| flag.strip_prefix("-C"))
        {
            options.push(option);
        }
    }

    options
}

fn target_cpu_of(encoded_rustflags: &str) -> Option<String> {
    codegen_options(encoded_rustflags)
        .into_iter()
        .rev()
        .find_map(|option| option.strip_prefix("target-cpu="))
        .map(ToOwned::to_owned)
}

fn map_cpu_feature_to_ggml(feature: &str) -> Option<&'static str> {
    match feature {
        "avx" => Some("GGML_AVX"),
        "avx2" => Some("GGML_AVX2"),
        "avx512bf16" => Some("GGML_AVX512_BF16"),
        "avx512vbmi" => Some("GGML_AVX512_VBMI"),
        "avx512vnni" => Some("GGML_AVX512_VNNI"),
        "avxvnni" => Some("GGML_AVX_VNNI"),
        "bmi2" => Some("GGML_BMI2"),
        "f16c" => Some("GGML_F16C"),
        "fma" => Some("GGML_FMA"),
        "sse4.2" => Some("GGML_SSE42"),
        _ => None,
    }
}

/// How ggml is compiled for the CPU rustc compiles for, and the debug messages explaining it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuTuning {
    pub defines: Vec<CmakeDefine>,
    pub compiler_flags: Vec<String>,
    pub debug_messages: Vec<CargoDirective>,
}

impl CpuTuning {
    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when the rustflags or the target features are not
    /// Unicode.
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        let target_cpu = environment
            .optional("CARGO_ENCODED_RUSTFLAGS")?
            .and_then(|rustflags| target_cpu_of(&rustflags));

        if target_cpu.as_deref() == Some("native") {
            return Ok(Self {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "ON")],
                compiler_flags: Vec::new(),
                debug_messages: vec![CargoDirective::Debug(
                    "Detected target-cpu=native, compiling with GGML_NATIVE".to_owned(),
                )],
            });
        }

        let target_features = environment
            .optional("CARGO_CFG_TARGET_FEATURE")?
            .unwrap_or_default();
        let compiler_flags = target_cpu
            .iter()
            .map(|cpu| format!("-march={cpu}"))
            .collect::<Vec<String>>();
        let mut defines = vec![CmakeDefine::new("GGML_NATIVE", "OFF")];

        defines.extend(
            target_features
                .split(',')
                .filter_map(map_cpu_feature_to_ggml)
                .map(|ggml_switch| CmakeDefine::new(ggml_switch, "ON")),
        );

        if context.cargo_cfg_target_arch == "aarch64" && context.target_os == TargetOs::Linux {
            defines.push(CmakeDefine::new("GGML_CPU_ARM_ARCH", "armv8-a"));
        }

        Ok(Self {
            defines,
            debug_messages: compiler_flags
                .iter()
                .map(|flag| format!("Setting baseline architecture: {flag}"))
                .chain([format!("Compiling with target features: {target_features}")])
                .map(CargoDirective::Debug)
                .collect(),
            compiler_flags,
        })
    }
}

#[cfg(test)]
mod cpu_feature_mapping_tests {
    use super::map_cpu_feature_to_ggml;

    #[test]
    fn every_supported_rust_cpu_feature_maps_to_its_ggml_switch() {
        assert_eq!(map_cpu_feature_to_ggml("avx"), Some("GGML_AVX"));
        assert_eq!(map_cpu_feature_to_ggml("avx2"), Some("GGML_AVX2"));
        assert_eq!(
            map_cpu_feature_to_ggml("avx512bf16"),
            Some("GGML_AVX512_BF16")
        );
        assert_eq!(
            map_cpu_feature_to_ggml("avx512vbmi"),
            Some("GGML_AVX512_VBMI")
        );
        assert_eq!(
            map_cpu_feature_to_ggml("avx512vnni"),
            Some("GGML_AVX512_VNNI")
        );
        assert_eq!(map_cpu_feature_to_ggml("avxvnni"), Some("GGML_AVX_VNNI"));
        assert_eq!(map_cpu_feature_to_ggml("bmi2"), Some("GGML_BMI2"));
        assert_eq!(map_cpu_feature_to_ggml("f16c"), Some("GGML_F16C"));
        assert_eq!(map_cpu_feature_to_ggml("fma"), Some("GGML_FMA"));
        assert_eq!(map_cpu_feature_to_ggml("sse4.2"), Some("GGML_SSE42"));
    }

    #[test]
    fn unsupported_rust_cpu_feature_does_not_enable_a_ggml_switch() {
        assert_eq!(map_cpu_feature_to_ggml("aes"), None);
        assert_eq!(map_cpu_feature_to_ggml(""), None);
    }
}

#[cfg(test)]
mod target_cpu_of_tests {
    use super::target_cpu_of;

    #[test]
    fn an_attached_codegen_target_cpu_is_found() {
        assert_eq!(
            target_cpu_of("-Ctarget-cpu=native"),
            Some("native".to_owned())
        );
    }

    #[test]
    fn a_codegen_target_cpu_given_as_the_next_flag_is_found() {
        assert_eq!(
            [
                target_cpu_of("-C\x1ftarget-cpu=znver4"),
                target_cpu_of("--codegen\x1ftarget-cpu=znver4"),
                target_cpu_of("--codegen=target-cpu=znver4"),
            ],
            [
                Some("znver4".to_owned()),
                Some("znver4".to_owned()),
                Some("znver4".to_owned()),
            ]
        );
    }

    #[test]
    fn the_last_target_cpu_wins_as_it_does_for_rustc() {
        assert_eq!(
            target_cpu_of("-Ctarget-cpu=haswell\x1f-Ctarget-cpu=znver4"),
            Some("znver4".to_owned())
        );
    }

    #[test]
    fn a_target_cpu_inside_another_codegen_option_is_not_the_target_cpu() {
        assert_eq!(target_cpu_of("-Clink-arg=-Wl,target-cpu=znver4"), None);
    }

    #[test]
    fn rustflags_without_a_target_cpu_name_no_target_cpu() {
        assert_eq!(
            target_cpu_of("-Copt-level=3\x1f--cfg\x1ffeature=\"x\""),
            None
        );
    }
}

#[cfg(test)]
mod cpu_tuning_tests {
    use enumflags2::BitFlags;

    use super::CpuTuning;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_environment::tests::with_non_unicode;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_define::CmakeDefine;

    fn tuned(overrides: &[(&str, &str)]) -> CpuTuning {
        let environment = cargo_environment(BitFlags::empty(), overrides);

        CpuTuning::of(&context_of(&environment), &environment).unwrap()
    }

    #[test]
    fn a_native_target_cpu_lets_ggml_detect_the_cpu() {
        assert_eq!(
            tuned(&[
                ("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native"),
                ("CARGO_CFG_TARGET_FEATURE", "avx2"),
            ]),
            CpuTuning {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "ON")],
                compiler_flags: Vec::new(),
                debug_messages: vec![CargoDirective::Debug(
                    "Detected target-cpu=native, compiling with GGML_NATIVE".to_owned()
                )],
            }
        );
    }

    #[test]
    fn a_named_target_cpu_sets_the_baseline_and_target_features_enable_their_ggml_switches() {
        assert_eq!(
            tuned(&[
                ("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=znver4"),
                ("CARGO_CFG_TARGET_FEATURE", "aes,avx2,fma"),
            ]),
            CpuTuning {
                defines: vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("GGML_AVX2", "ON"),
                    CmakeDefine::new("GGML_FMA", "ON"),
                ],
                compiler_flags: vec!["-march=znver4".to_owned()],
                debug_messages: vec![
                    CargoDirective::Debug(
                        "Setting baseline architecture: -march=znver4".to_owned()
                    ),
                    CargoDirective::Debug(
                        "Compiling with target features: aes,avx2,fma".to_owned()
                    ),
                ],
            }
        );
    }

    #[test]
    fn aarch64_linux_without_a_target_cpu_builds_for_the_armv8_baseline() {
        assert_eq!(
            tuned(&[
                ("CARGO_CFG_TARGET_ARCH", "aarch64"),
                ("TARGET", "aarch64-unknown-linux-gnu"),
            ]),
            CpuTuning {
                defines: vec![
                    CmakeDefine::new("GGML_NATIVE", "OFF"),
                    CmakeDefine::new("GGML_CPU_ARM_ARCH", "armv8-a"),
                ],
                compiler_flags: Vec::new(),
                debug_messages: vec![CargoDirective::Debug(
                    "Compiling with target features: ".to_owned()
                )],
            }
        );
    }

    #[test]
    fn rustflags_or_target_features_that_are_not_unicode_are_refused() {
        let context = context_of(&cargo_environment(BitFlags::empty(), &[]));

        assert_eq!(
            ["CARGO_ENCODED_RUSTFLAGS", "CARGO_CFG_TARGET_FEATURE"].map(|name| {
                CpuTuning::of(
                    &context,
                    &with_non_unicode(cargo_environment(BitFlags::empty(), &[]), name),
                )
                .err()
                .map(|error| error.to_string())
            }),
            ["CARGO_ENCODED_RUSTFLAGS", "CARGO_CFG_TARGET_FEATURE"].map(|name| Some(format!(
                "environment variable {name} could not be read: environment variable was not \
                 valid unicode: \"\\xFF\""
            )))
        );
    }
}
