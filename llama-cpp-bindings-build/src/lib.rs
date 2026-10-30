#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

mod android_architecture;
mod android_ndk;
mod binding_callbacks;
mod bindings_generation;
mod build_context;
pub mod build_environment;
pub mod build_error;
pub mod build_feature;
mod cargo_directive;
mod cmake_build;
mod cmake_configuration;
mod cmake_define;
mod cpu_tuning;
mod emit_directives;
mod enabled_build_features;
mod host_os;
mod link_directives;
mod link_kind;
mod native_build;
mod native_library;
mod privatized_field;
mod rebuild_triggers;
mod target_os;
mod wrapper_compilation;

use std::io::Write;

use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::host_os::HostOs;
use crate::native_build::NativeBuild;

/// Generates the bindings, compiles the C++ wrappers, and builds llama.cpp for the build script
/// run `environment` describes, then writes the cargo directives that link them to `output`.
///
/// # Errors
///
/// Returns the [`BuildError`] of the first decision or build step that fails; nothing is written
/// to `output` then.
pub fn build(environment: &BuildEnvironment, output: &mut impl Write) -> Result<(), BuildError> {
    let context = BuildContext::from_environment(environment, HostOs::current())?;

    NativeBuild::of(&context, environment)?.run(output)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use enumflags2::BitFlags;

    use super::build;
    use crate::build_context::tests::cargo_environment;
    use crate::build_environment::BuildEnvironment;
    use crate::build_feature::BuildFeature;

    const FIXTURE_CMAKE_PROJECT: &str =
        concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/cmake_project");

    fn failure(environment: &BuildEnvironment) -> [Option<String>; 2] {
        let mut output = Vec::new();

        [
            build(environment, &mut output)
                .err()
                .map(|error| error.to_string()),
            String::from_utf8(output).ok(),
        ]
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
                    &[("CARGO_MANIFEST_DIR", FIXTURE_CMAKE_PROJECT)]
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
