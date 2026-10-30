use llama_cpp_wrapper_sources::wrapper_headers::WRAPPER_HEADERS;
use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

use crate::build_context::BuildContext;
use crate::cargo_directive::CargoDirective;

/// The files and variables whose change makes cargo run the native build again.
#[must_use]
pub fn rebuild_triggers(context: &BuildContext) -> Vec<CargoDirective> {
    [
        CargoDirective::RerunIfChanged(context.sys_dir.join("build.rs")),
        CargoDirective::RerunIfEnvChanged("CUDAARCHS".to_owned()),
    ]
    .into_iter()
    .chain(
        WRAPPER_HEADERS
            .iter()
            .chain(WRAPPER_SOURCES)
            .map(|wrapper_file| CargoDirective::RerunIfChanged(context.sys_dir.join(wrapper_file))),
    )
    .chain([CargoDirective::RerunIfChanged(context.llama_src())])
    .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use enumflags2::BitFlags;
    use llama_cpp_wrapper_sources::wrapper_headers::WRAPPER_HEADERS;
    use llama_cpp_wrapper_sources::wrapper_sources::WRAPPER_SOURCES;

    use super::rebuild_triggers;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::cargo_directive::CargoDirective;

    #[test]
    fn the_build_script_cuda_architectures_every_wrapper_file_and_llama_cpp_trigger_a_rebuild() {
        let triggers = rebuild_triggers(&context_of(&cargo_environment(BitFlags::empty(), &[])));

        assert_eq!(
            triggers,
            [
                vec![
                    CargoDirective::RerunIfChanged(PathBuf::from("/sys/build.rs")),
                    CargoDirective::RerunIfEnvChanged("CUDAARCHS".to_owned()),
                ],
                WRAPPER_HEADERS
                    .iter()
                    .chain(WRAPPER_SOURCES)
                    .map(|wrapper_file| {
                        CargoDirective::RerunIfChanged(PathBuf::from(format!(
                            "/sys/{wrapper_file}"
                        )))
                    })
                    .collect(),
                vec![CargoDirective::RerunIfChanged(PathBuf::from(
                    "/sys/llama.cpp"
                ))],
            ]
            .concat()
        );
    }
}
