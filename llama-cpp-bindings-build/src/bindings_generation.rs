use std::path::PathBuf;

use bindgen::Builder;
use llama_cpp_wrapper_sources::fault_injection_define::FAULT_INJECTION_DEFINE;

use crate::binding_callbacks::BindingCallbacks;
use crate::build_context::BuildContext;
use crate::build_environment::BuildEnvironment;
use crate::build_error::BuildError;
use crate::build_feature::BuildFeature;
use crate::cargo_directive::CargoDirective;
use crate::target_os::TargetOs;

const DEPRECATED_FUNCTIONS: &[&str] = &[
    "ggml_add1",
    "ggml_add1_inplace",
    "ggml_flash_attn_ext_set_prec",
    "ggml_mul_mat_set_prec",
    "ggml_rope_custom",
    "ggml_rope_custom_inplace",
    "ggml_type_sizef",
    "ggml_upscale_ext",
    "llama_add_bos_token",
    "llama_add_eos_token",
    "llama_copy_state_data",
    "llama_free_model",
    "llama_get_state_size",
    "llama_load_model_from_file",
    "llama_load_session_file",
    "llama_n_ctx_train",
    "llama_n_embd",
    "llama_n_head",
    "llama_n_layer",
    "llama_n_vocab",
    "llama_new_context_with_model",
    "llama_sampler_init_grammar_lazy",
    "llama_save_session_file",
    "llama_set_state_data",
    "llama_set_warmup",
    "llama_token_bos",
    "llama_token_cls",
    "llama_token_eos",
    "llama_token_eot",
    "llama_token_fim_mid",
    "llama_token_fim_pad",
    "llama_token_fim_pre",
    "llama_token_fim_rep",
    "llama_token_fim_sep",
    "llama_token_fim_suf",
    "llama_token_get_attr",
    "llama_token_get_score",
    "llama_token_get_text",
    "llama_token_is_control",
    "llama_token_is_eog",
    "llama_token_nl",
    "llama_token_pad",
    "llama_token_sep",
    "llama_vocab_cls",
    "mtmd_encode",
    "mtmd_helper_model_can_chat",
    "mtmd_helper_support_video",
    "mtmd_image_tokens_get_nx",
    "mtmd_image_tokens_get_ny",
];

fn configured_builder() -> Builder {
    let builder = Builder::default()
        .generate_comments(false)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_var("GGML_MAX_DIMS")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_TOKEN_NULL")
        .allowlist_var("LLAMA_DRY_MAX_SEQ_BREAKERS")
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .blocklist_function("ggml_fopen")
        .blocklist_function("llama_adapter_lora_init_from_file_ptr")
        .blocklist_function("gguf_init_from_file_ptr")
        .blocklist_function("gguf_write_to_file_ptr")
        .blocklist_function("llama_model_load_from_file_ptr")
        .blocklist_type("FILE")
        .blocklist_type("_IO_.*")
        .prepend_enum_name(false);

    DEPRECATED_FUNCTIONS
        .iter()
        .fold(builder, |builder, function| {
            builder.blocklist_function(function)
        })
}

fn fault_injection_args(environment: &BuildEnvironment) -> Vec<String> {
    if environment.has(BuildFeature::FaultInjection) {
        vec![format!("-D{FAULT_INJECTION_DEFINE}")]
    } else {
        Vec::new()
    }
}

fn target_args(context: &BuildContext) -> Vec<String> {
    match &context.target_os {
        TargetOs::Android(android_ndk) => android_ndk.clang_args(&context.target_triple),
        TargetOs::Apple | TargetOs::Linux => Vec::new(),
    }
}

/// Generates the Rust bindings of `header` with bindgen into `output`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BindingsGeneration {
    pub header: PathBuf,
    pub clang_args: Vec<String>,
    pub output: PathBuf,
}

impl BindingsGeneration {
    #[must_use]
    pub fn of(context: &BuildContext, environment: &BuildEnvironment) -> Self {
        let llama_src = context.llama_src();

        Self {
            header: context.sys_dir.join("wrapper.h"),
            clang_args: [
                vec![
                    format!("-I{}", llama_src.display()),
                    format!("-I{}", llama_src.join("include").display()),
                    format!("-I{}", llama_src.join("ggml/include").display()),
                ],
                fault_injection_args(environment),
                target_args(context),
            ]
            .concat(),
            output: context.out_dir.join("bindings.rs"),
        }
    }

    /// Writes the bindings and returns the cargo directives that rebuild them when what they
    /// were generated from changes.
    ///
    /// # Errors
    ///
    /// Returns [`BuildError::Bindgen`] when bindgen cannot parse the header,
    /// [`BuildError::PrivatizedFieldMissing`] when a field the bindings keep private no longer
    /// exists, or [`BuildError::BindingsWrite`] when `output` cannot be written.
    pub fn generate(&self) -> Result<Vec<CargoDirective>, BuildError> {
        let callbacks = BindingCallbacks::default();
        let bindings = configured_builder()
            .header(self.header.display().to_string())
            .clang_args(&self.clang_args)
            .parse_callbacks(Box::new(callbacks.clone()))
            .generate()
            .map_err(BuildError::Bindgen)?;

        callbacks.verify_every_privatized_field_was_found()?;
        bindings
            .write_to_file(&self.output)
            .map_err(BuildError::BindingsWrite)?;

        Ok([
            callbacks.take_rebuild_triggers(),
            vec![CargoDirective::Debug("Bindings Created".to_owned())],
        ]
        .concat())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use enumflags2::BitFlags;
    use llama_cpp_wrapper_sources::fault_injection_define::FAULT_INJECTION_DEFINE;

    use super::BindingsGeneration;
    use crate::android_ndk::tests::FIXTURE_ANDROID_NDK;
    use crate::build_context::tests::ANDROID_OVERRIDES;
    use crate::build_context::tests::cargo_environment;
    use crate::build_context::tests::context_of;
    use crate::build_feature::BuildFeature;

    const FIXTURE_BINDINGS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/bindings");

    fn planned(features: BitFlags<BuildFeature>, overrides: &[(&str, &str)]) -> BindingsGeneration {
        let environment = cargo_environment(features, overrides);

        BindingsGeneration::of(&context_of(&environment), &environment)
    }

    fn llama_cpp_includes() -> Vec<String> {
        vec![
            "-I/sys/llama.cpp".to_owned(),
            "-I/sys/llama.cpp/include".to_owned(),
            "-I/sys/llama.cpp/ggml/include".to_owned(),
        ]
    }

    fn generation_failure(header: &str, output: PathBuf) -> Option<String> {
        BindingsGeneration {
            header: PathBuf::from(FIXTURE_BINDINGS).join(header),
            clang_args: Vec::new(),
            output,
        }
        .generate()
        .err()
        .map(|error| error.to_string())
    }

    #[test]
    fn the_wrapper_header_is_parsed_against_the_llama_cpp_headers() {
        assert_eq!(
            planned(BitFlags::empty(), &[]),
            BindingsGeneration {
                header: PathBuf::from("/sys/wrapper.h"),
                clang_args: llama_cpp_includes(),
                output: PathBuf::from("/out/bindings.rs"),
            }
        );
    }

    #[test]
    fn fault_injection_and_android_targets_add_their_clang_arguments() {
        let toolchain = format!("{FIXTURE_ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64");

        assert_eq!(
            planned(
                BitFlags::from(BuildFeature::FaultInjection),
                &ANDROID_OVERRIDES
            )
            .clang_args,
            [
                llama_cpp_includes(),
                vec![
                    format!("-D{FAULT_INJECTION_DEFINE}"),
                    format!("--sysroot={toolchain}/sysroot"),
                    "-D__ANDROID_API__=28".to_owned(),
                    "-D__ANDROID__".to_owned(),
                    "-isystem".to_owned(),
                    format!("{toolchain}/lib/clang/20/include"),
                    "-isystem".to_owned(),
                    format!("{toolchain}/sysroot/usr/include/aarch64-linux-android"),
                    "-isystem".to_owned(),
                    format!("{toolchain}/sysroot/usr/include"),
                    "-include".to_owned(),
                    "stdbool.h".to_owned(),
                    "-include".to_owned(),
                    "stdint.h".to_owned(),
                    "--target=aarch64-linux-android".to_owned(),
                ],
            ]
            .concat()
        );
    }

    #[test]
    fn a_missing_header_is_reported_by_bindgen() {
        assert_eq!(
            generation_failure("missing.h", PathBuf::from("/out/bindings.rs")),
            Some(format!(
                "bindgen failed: header '{FIXTURE_BINDINGS}/missing.h' does not exist."
            ))
        );
    }

    #[test]
    fn a_header_without_a_privatized_field_is_refused() {
        assert_eq!(
            generation_failure(
                "missing_privatized_field.h",
                PathBuf::from("/out/bindings.rs")
            ),
            Some(
                "bindgen never saw field llama_context_params::defrag_thold, so it can no longer \
                 be kept private"
                    .to_owned()
            )
        );
    }

    #[test]
    fn bindings_that_cannot_be_written_are_reported() {
        assert_eq!(
            generation_failure(
                "wrapper.h",
                PathBuf::from(FIXTURE_BINDINGS).join("missing-directory/bindings.rs")
            ),
            Some(
                "generated bindings could not be written: No such file or directory (os error 2)"
                    .to_owned()
            )
        );
    }
}
