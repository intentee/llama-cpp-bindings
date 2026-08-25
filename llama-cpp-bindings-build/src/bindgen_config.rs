use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use crate::BuildError;
use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::target_os::TargetOs;

const DEPRECATED_FUNCTIONS: &[&str] = &[
    "ggml_add1",
    "ggml_add1_inplace",
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
    "mtmd_image_tokens_get_nx",
    "mtmd_image_tokens_get_ny",
];

struct PrivatizedField {
    type_name: &'static str,
    field_name: &'static str,
}

const PRIVATIZED_FIELDS: &[PrivatizedField] = &[
    PrivatizedField {
        type_name: "llama_context_params",
        field_name: "defrag_thold",
    },
    PrivatizedField {
        type_name: "mtmd_context_params",
        field_name: "image_marker",
    },
];

#[derive(Clone, Debug)]
struct BindingCallbacks {
    privatized_field_hits: Arc<Vec<AtomicBool>>,
}

impl BindingCallbacks {
    fn new() -> Self {
        Self {
            privatized_field_hits: Arc::new(
                PRIVATIZED_FIELDS
                    .iter()
                    .map(|_| AtomicBool::new(false))
                    .collect(),
            ),
        }
    }

    fn verify_every_privatized_field_was_found(&self) -> Result<(), BuildError> {
        for (field, was_found) in PRIVATIZED_FIELDS
            .iter()
            .zip(self.privatized_field_hits.iter())
        {
            if !was_found.load(Ordering::Relaxed) {
                return Err(BuildError::PrivatizedFieldMissing {
                    type_name: field.type_name,
                    field_name: field.field_name,
                });
            }
        }

        Ok(())
    }
}

impl bindgen::callbacks::ParseCallbacks for BindingCallbacks {
    fn header_file(&self, filename: &str) {
        println!("cargo:rerun-if-changed={filename}");
    }

    fn include_file(&self, filename: &str) {
        println!("cargo:rerun-if-changed={filename}");
    }

    fn read_env_var(&self, key: &str) {
        println!("cargo:rerun-if-env-changed={key}");
    }

    fn field_visibility(
        &self,
        info: bindgen::callbacks::FieldInfo<'_>,
    ) -> Option<bindgen::FieldVisibilityKind> {
        for (field, was_found) in PRIVATIZED_FIELDS
            .iter()
            .zip(self.privatized_field_hits.iter())
        {
            if field.type_name == info.type_name && field.field_name == info.field_name {
                was_found.store(true, Ordering::Relaxed);

                return Some(bindgen::FieldVisibilityKind::Private);
            }
        }

        None
    }
}

pub fn generate_bindings(
    llama_src: &Path,
    out_dir: &Path,
    target_os: TargetOs,
    target_triple: &str,
    android_ndk: Option<&AndroidNdk>,
) -> Result<(), BuildError> {
    let callbacks = BindingCallbacks::new();
    let mut builder = create_base_builder(llama_src, callbacks.clone());

    if target_os.is_android()
        && let Some(ndk) = android_ndk
    {
        builder = configure_android_bindgen(builder, ndk, target_triple);
    }

    if target_os.is_msvc() {
        builder = configure_msvc_bindgen(builder, target_triple)?;
    }

    let bindings = builder.generate().map_err(BuildError::Bindgen)?;

    callbacks.verify_every_privatized_field_was_found()?;

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .map_err(BuildError::BindingsWrite)?;

    debug_log!("Bindings Created");

    Ok(())
}

fn create_base_builder(llama_src: &Path, callbacks: BindingCallbacks) -> bindgen::Builder {
    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .parse_callbacks(Box::new(callbacks))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .blocklist_function("ggml_fopen")
        .blocklist_function("gguf_init_from_file_ptr")
        .blocklist_function("gguf_write_to_file_ptr")
        .blocklist_function("llama_model_load_from_file_ptr")
        .blocklist_type("FILE")
        .blocklist_type("_IO_.*")
        .blocklist_type("_iobuf")
        .prepend_enum_name(false);

    for function in DEPRECATED_FUNCTIONS {
        builder = builder.blocklist_function(function);
    }

    builder
}

fn configure_android_bindgen(
    mut builder: bindgen::Builder,
    ndk: &AndroidNdk,
    target_triple: &str,
) -> bindgen::Builder {
    builder = builder
        .clang_arg(format!("--sysroot={}", ndk.sysroot))
        .clang_arg(format!("-D__ANDROID_API__={}", ndk.api_level))
        .clang_arg("-D__ANDROID__");

    builder = builder
        .clang_arg("-isystem")
        .clang_arg(&ndk.clang_builtin_includes);

    builder = builder
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include/{}", ndk.sysroot, ndk.target_prefix))
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include", ndk.sysroot))
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        .clang_arg("-include")
        .clang_arg("stdint.h");

    builder.clang_arg(format!("--target={target_triple}"))
}

fn configure_msvc_bindgen(
    mut builder: bindgen::Builder,
    target_triple: &str,
) -> Result<bindgen::Builder, BuildError> {
    let compiler = cc::Build::new()
        .try_get_compiler()
        .map_err(BuildError::NativeCompiler)?;

    let msvc_include_paths = compiler
        .env()
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("INCLUDE"))
        .map(|(_, value)| value.clone());

    if let Some(include_paths) = msvc_include_paths {
        for include_path in include_paths
            .to_string_lossy()
            .split(';')
            .filter(|path| !path.is_empty())
        {
            builder = builder.clang_arg("-isystem").clang_arg(include_path);
            debug_log!("Added MSVC include path: {}", include_path);
        }
    }

    builder = builder
        .clang_arg(format!("--target={target_triple}"))
        .clang_arg("-fms-compatibility")
        .clang_arg("-fms-extensions");

    debug_log!(
        "Configured bindgen with MSVC toolchain for target: {}",
        target_triple
    );

    Ok(builder)
}
