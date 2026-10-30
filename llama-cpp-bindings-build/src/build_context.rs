use std::path::PathBuf;

use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cargo_directive::CargoDirective;
use crate::host_os::HostOs;
use crate::target_os::TargetOs;

fn native_profile(cargo_profile: &str) -> String {
    match cargo_profile {
        "debug" | "release" => "Release".to_owned(),
        other => other.to_owned(),
    }
}

/// Where and for what the native libraries are built, as cargo describes the build script run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BuildContext {
    pub out_dir: PathBuf,
    pub sys_dir: PathBuf,
    pub target_os: TargetOs,
    pub target_triple: String,
    pub host_triple: String,
    pub cargo_cfg_target_arch: String,
    pub cargo_cfg_target_env: String,
    pub build_shared_libs: bool,
    pub profile: String,
    pub opt_level: String,
    pub debug_info: String,
    pub include_debug_messages: bool,
}

impl BuildContext {
    /// # Errors
    ///
    /// Returns [`BuildError::Environment`] for a variable cargo sets for build scripts that is
    /// missing or not Unicode, or the [`BuildError`] of [`TargetOs::detect`] when the target
    /// cannot be built for.
    pub fn from_environment(
        environment: &BuildEnvironment,
        host_os: HostOs,
    ) -> Result<Self, BuildError> {
        let target_triple = environment.required("TARGET")?;
        let cargo_cfg_target_arch = environment.required("CARGO_CFG_TARGET_ARCH")?;

        Ok(Self {
            out_dir: PathBuf::from(environment.required("OUT_DIR")?),
            sys_dir: PathBuf::from(environment.required("CARGO_MANIFEST_DIR")?),
            target_os: TargetOs::detect(
                environment,
                &target_triple,
                &cargo_cfg_target_arch,
                host_os,
            )?,
            host_triple: environment.required("HOST")?,
            cargo_cfg_target_env: environment
                .optional("CARGO_CFG_TARGET_ENV")?
                .unwrap_or_default(),
            build_shared_libs: environment.has(BuildFeature::DynamicLink),
            profile: native_profile(&environment.required("PROFILE")?),
            opt_level: environment.required("OPT_LEVEL")?,
            debug_info: environment.required("DEBUG")?,
            include_debug_messages: environment.optional("BUILD_DEBUG")?.is_some(),
            target_triple,
            cargo_cfg_target_arch,
        })
    }

    #[must_use]
    pub fn cmake_dir(&self) -> PathBuf {
        self.out_dir.join("cmake")
    }

    #[must_use]
    pub fn llama_src(&self) -> PathBuf {
        self.sys_dir.join("llama.cpp")
    }

    #[must_use]
    pub fn debug_messages(&self) -> Vec<CargoDirective> {
        [
            format!("TARGET: {}", self.target_triple),
            format!("CARGO_MANIFEST_DIR: {}", self.sys_dir.display()),
            format!("OUT_DIR: {}", self.out_dir.display()),
            format!("CMAKE_DIR: {}", self.cmake_dir().display()),
            format!("BUILD_SHARED: {}", self.build_shared_libs),
        ]
        .map(CargoDirective::Debug)
        .to_vec()
    }
}

#[cfg(test)]
pub mod tests {
    use std::ffi::OsStr;
    use std::ffi::OsString;
    use std::path::PathBuf;

    use enumflags2::BitFlags;

    use super::BuildContext;
    use super::native_profile;
    use crate::android_ndk::tests::FIXTURE_ANDROID_NDK;
    use crate::build_environment::BuildEnvironment;
    use crate::build_environment::tests::environment_of;
    use crate::build_environment::tests::with_non_unicode;
    use crate::build_feature::BuildFeature;
    use crate::cargo_directive::CargoDirective;
    use crate::host_os::HostOs;
    use crate::target_os::TargetOs;

    const CARGO_VARIABLES: [(&str, &str); 10] = [
        ("CARGO_CFG_TARGET_ARCH", "x86_64"),
        ("CARGO_CFG_TARGET_ENV", "gnu"),
        ("CARGO_CFG_TARGET_OS", "linux"),
        ("CARGO_MANIFEST_DIR", "/sys"),
        ("DEBUG", "false"),
        ("HOST", "x86_64-unknown-linux-gnu"),
        ("OPT_LEVEL", "3"),
        ("OUT_DIR", "/out"),
        ("PROFILE", "release"),
        ("TARGET", "x86_64-unknown-linux-gnu"),
    ];

    pub const ANDROID_OVERRIDES: [(&str, &str); 4] = [
        ("ANDROID_NDK_HOME", FIXTURE_ANDROID_NDK),
        ("CARGO_CFG_TARGET_ARCH", "aarch64"),
        ("CARGO_CFG_TARGET_OS", "android"),
        ("TARGET", "aarch64-linux-android"),
    ];

    /// The variables cargo sets for a build script building for x86-64 Linux, with `overrides`.
    pub fn cargo_environment(
        features: BitFlags<BuildFeature>,
        overrides: &[(&str, &str)],
    ) -> BuildEnvironment {
        let mut environment = environment_of(
            features,
            CARGO_VARIABLES.map(|(name, value)| (name, OsString::from(value))),
        );

        environment.variables.extend(
            overrides
                .iter()
                .map(|(name, value)| (OsString::from(name), OsString::from(value))),
        );

        environment
    }

    pub fn context_of(environment: &BuildEnvironment) -> BuildContext {
        BuildContext::from_environment(environment, HostOs::Linux).unwrap()
    }

    #[test]
    fn cargo_debug_and_release_profiles_both_use_optimized_native_code() {
        assert_eq!(native_profile("debug"), "Release");
        assert_eq!(native_profile("release"), "Release");
    }

    #[test]
    fn custom_cargo_profile_name_is_preserved_for_cmake() {
        assert_eq!(native_profile("RelWithDebInfo"), "RelWithDebInfo");
    }

    #[test]
    fn the_cargo_variables_and_features_describe_the_build() {
        assert_eq!(
            context_of(&cargo_environment(
                BitFlags::from(BuildFeature::DynamicLink),
                &[("BUILD_DEBUG", "")]
            )),
            BuildContext {
                out_dir: PathBuf::from("/out"),
                sys_dir: PathBuf::from("/sys"),
                target_os: TargetOs::Linux,
                target_triple: "x86_64-unknown-linux-gnu".to_owned(),
                host_triple: "x86_64-unknown-linux-gnu".to_owned(),
                cargo_cfg_target_arch: "x86_64".to_owned(),
                cargo_cfg_target_env: "gnu".to_owned(),
                build_shared_libs: true,
                profile: "Release".to_owned(),
                opt_level: "3".to_owned(),
                debug_info: "false".to_owned(),
                include_debug_messages: true,
            }
        );
    }

    #[test]
    fn every_variable_cargo_sets_for_build_scripts_is_required() {
        let without = |name: &str| {
            let mut environment = cargo_environment(BitFlags::empty(), &[]);

            environment.variables.remove(OsStr::new(name));

            BuildContext::from_environment(&environment, HostOs::Linux)
                .err()
                .map(|error| error.to_string())
        };
        let required = CARGO_VARIABLES
            .map(|(name, _)| name)
            .into_iter()
            .filter(|name| *name != "CARGO_CFG_TARGET_ENV")
            .collect::<Vec<&str>>();

        assert_eq!(
            required
                .iter()
                .map(|name| without(name))
                .collect::<Vec<Option<String>>>(),
            required
                .iter()
                .map(|name| Some(format!(
                    "environment variable {name} could not be read: environment variable not \
                     found"
                )))
                .collect::<Vec<Option<String>>>()
        );
    }

    #[test]
    fn optional_variables_that_are_not_unicode_are_refused() {
        assert_eq!(
            ["CARGO_CFG_TARGET_ENV", "BUILD_DEBUG"].map(|name| {
                BuildContext::from_environment(
                    &with_non_unicode(cargo_environment(BitFlags::empty(), &[]), name),
                    HostOs::Linux,
                )
                .err()
                .map(|error| error.to_string())
            }),
            ["CARGO_CFG_TARGET_ENV", "BUILD_DEBUG"].map(|name| Some(format!(
                "environment variable {name} could not be read: environment variable was not \
                 valid unicode: \"\\xFF\""
            )))
        );
    }

    #[test]
    fn debug_messages_describe_where_the_build_happens() {
        assert_eq!(
            context_of(&cargo_environment(BitFlags::empty(), &[])).debug_messages(),
            [
                "TARGET: x86_64-unknown-linux-gnu",
                "CARGO_MANIFEST_DIR: /sys",
                "OUT_DIR: /out",
                "CMAKE_DIR: /out/cmake",
                "BUILD_SHARED: false",
            ]
            .map(|message| CargoDirective::Debug(message.to_owned()))
        );
    }
}
