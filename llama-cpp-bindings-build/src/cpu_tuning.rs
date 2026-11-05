use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cargo_directive::CargoDirective;
use crate::cmake_define::CmakeDefine;
use crate::x86_ggml_switch::X86GgmlSwitch;

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

const X86_GGML_SWITCHES: [X86GgmlSwitch; 14] = [
    X86GgmlSwitch {
        name: "GGML_SSE42",
        target_features: &["sse4.2"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX",
        target_features: &["avx"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX_VNNI",
        target_features: &["avxvnni"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX2",
        target_features: &["avx2"],
    },
    X86GgmlSwitch {
        name: "GGML_BMI2",
        target_features: &["bmi2"],
    },
    X86GgmlSwitch {
        name: "GGML_FMA",
        target_features: &["fma"],
    },
    X86GgmlSwitch {
        name: "GGML_F16C",
        target_features: &["f16c"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX512",
        target_features: &["avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX512_VBMI",
        target_features: &["avx512vbmi"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX512_VNNI",
        target_features: &["avx512vnni"],
    },
    X86GgmlSwitch {
        name: "GGML_AVX512_BF16",
        target_features: &["avx512bf16"],
    },
    X86GgmlSwitch {
        name: "GGML_AMX_TILE",
        target_features: &["amx-tile"],
    },
    X86GgmlSwitch {
        name: "GGML_AMX_INT8",
        target_features: &["amx-int8"],
    },
    X86GgmlSwitch {
        name: "GGML_AMX_BF16",
        target_features: &["amx-bf16"],
    },
];

/// How ggml is compiled for the CPU rustc compiles for, and the debug messages explaining it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuTuning {
    pub defines: Vec<CmakeDefine>,
    pub debug_messages: Vec<CargoDirective>,
}

impl CpuTuning {
    /// ggml builds every CPU variant and selects one at runtime when backends load dynamically.
    /// Otherwise `target-cpu=native` lets ggml detect the CPU, and any other build switches
    /// every x86 instruction set explicitly on or off, the way rustc's target features describe
    /// the CPU, because ggml would enable AVX2-class defaults on its own.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] when the rustflags or the target features are not
    /// Unicode.
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Result<Self, BuildError> {
        if environment.has(BuildFeature::DynamicBackends) {
            return Ok(Self {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "OFF")],
                debug_messages: vec![CargoDirective::Debug(
                    "Building every CPU variant for ggml to select at runtime".to_owned(),
                )],
            });
        }

        let target_cpu = environment
            .optional("CARGO_ENCODED_RUSTFLAGS")?
            .and_then(|rustflags| target_cpu_of(&rustflags));

        if target_cpu.as_deref() == Some("native") {
            return Ok(Self {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "ON")],
                debug_messages: vec![CargoDirective::Debug(
                    "Detected target-cpu=native, compiling with GGML_NATIVE".to_owned(),
                )],
            });
        }

        let target_features = environment
            .optional("CARGO_CFG_TARGET_FEATURE")?
            .unwrap_or_default();
        let enabled_features = target_features.split(',').collect::<Vec<&str>>();
        let mut defines = vec![CmakeDefine::new("GGML_NATIVE", "OFF")];

        if matches!(context.cargo_cfg_target_arch.as_str(), "x86" | "x86_64") {
            defines.extend(
                X86_GGML_SWITCHES
                    .iter()
                    .map(|switch| switch.define_for(&enabled_features)),
            );
        }

        Ok(Self {
            defines,
            debug_messages: vec![CargoDirective::Debug(format!(
                "Compiling with target features: {target_features}"
            ))],
        })
    }
}

#[cfg(test)]
mod target_cpu_of_tests {
    use super::target_cpu_of;

    #[test]
    fn an_attached_codegen_target_cpu_is_found() {
        assert_eq!(
            [
                target_cpu_of("-Ctarget-cpu=native"),
                target_cpu_of("--codegen=target-cpu=znver4"),
            ],
            [Some("native".to_owned()), Some("znver4".to_owned())]
        );
    }

    #[test]
    fn a_codegen_target_cpu_given_as_the_next_flag_is_found() {
        assert_eq!(
            [
                target_cpu_of("-C\x1ftarget-cpu=znver4"),
                target_cpu_of("--codegen\x1ftarget-cpu=znver4"),
            ],
            [Some("znver4".to_owned()), Some("znver4".to_owned())]
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
    use crate::build_environment::tests::EnvironmentVariable;
    use crate::build_environment::tests::variable;
    use crate::build_environment::tests::with_non_unicode;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::cmake_define::CmakeDefine;

    fn tuned(features: BitFlags<BuildFeature>, overrides: &[EnvironmentVariable]) -> CpuTuning {
        let environment = cargo_environment(features, overrides);

        CpuTuning::of(&context_of(&environment), &environment).unwrap()
    }

    fn x86_switches(enabled: &[&'static str]) -> Vec<CmakeDefine> {
        [
            "GGML_SSE42",
            "GGML_AVX",
            "GGML_AVX_VNNI",
            "GGML_AVX2",
            "GGML_BMI2",
            "GGML_FMA",
            "GGML_F16C",
            "GGML_AVX512",
            "GGML_AVX512_VBMI",
            "GGML_AVX512_VNNI",
            "GGML_AVX512_BF16",
            "GGML_AMX_TILE",
            "GGML_AMX_INT8",
            "GGML_AMX_BF16",
        ]
        .into_iter()
        .map(|name| CmakeDefine::switch(name, enabled.contains(&name)))
        .collect()
    }

    #[test]
    fn a_native_target_cpu_lets_ggml_detect_the_cpu() {
        assert_eq!(
            tuned(
                BitFlags::empty(),
                &[
                    variable("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native"),
                    variable("CARGO_CFG_TARGET_FEATURE", "avx2"),
                ]
            ),
            CpuTuning {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "ON")],
                debug_messages: vec![CargoDirective::Debug(
                    "Detected target-cpu=native, compiling with GGML_NATIVE".to_owned()
                )],
            }
        );
    }

    #[test]
    fn a_baseline_x86_build_turns_every_instruction_set_ggml_would_assume_off() {
        assert_eq!(
            tuned(
                BitFlags::empty(),
                &[variable("CARGO_CFG_TARGET_FEATURE", "fxsr,sse,sse2")]
            ),
            CpuTuning {
                defines: [
                    vec![CmakeDefine::new("GGML_NATIVE", "OFF")],
                    x86_switches(&[]),
                ]
                .concat(),
                debug_messages: vec![CargoDirective::Debug(
                    "Compiling with target features: fxsr,sse,sse2".to_owned()
                )],
            }
        );
    }

    #[test]
    fn a_named_target_cpu_switches_on_exactly_the_instruction_sets_its_target_features_enable() {
        assert_eq!(
            tuned(
                BitFlags::empty(),
                &[
                    variable("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=znver4"),
                    variable(
                        "CARGO_CFG_TARGET_FEATURE",
                        "avx,avx2,avx512f,avx512cd,avx512vl,avx512dq,bmi2,f16c,fma,sse4.2"
                    ),
                ]
            )
            .defines,
            [
                vec![CmakeDefine::new("GGML_NATIVE", "OFF")],
                x86_switches(&[
                    "GGML_SSE42",
                    "GGML_AVX",
                    "GGML_AVX2",
                    "GGML_BMI2",
                    "GGML_FMA",
                    "GGML_F16C",
                ]),
            ]
            .concat()
        );
    }

    #[test]
    fn an_aarch64_build_leaves_the_architecture_to_the_compiler_baseline() {
        assert_eq!(
            tuned(
                BitFlags::empty(),
                &[
                    variable("CARGO_CFG_TARGET_ARCH", "aarch64"),
                    variable("TARGET", "aarch64-unknown-linux-gnu"),
                ]
            ),
            CpuTuning {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "OFF")],
                debug_messages: vec![CargoDirective::Debug(
                    "Compiling with target features: ".to_owned()
                )],
            }
        );
    }

    #[test]
    fn dynamic_backends_build_every_cpu_variant_without_tuning() {
        assert_eq!(
            tuned(
                BuildFeature::DynamicBackends | BuildFeature::DynamicLink,
                &[
                    variable("CARGO_CFG_TARGET_ARCH", "aarch64"),
                    variable("TARGET", "aarch64-unknown-linux-gnu"),
                    variable("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native"),
                ]
            ),
            CpuTuning {
                defines: vec![CmakeDefine::new("GGML_NATIVE", "OFF")],
                debug_messages: vec![CargoDirective::Debug(
                    "Building every CPU variant for ggml to select at runtime".to_owned()
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
