use std::env;
use std::path::Path;

use crate::android_ndk::AndroidNdk;
use crate::debug_log;
use crate::target_os::TargetOs;

pub fn generate_bindings(
    wrapper_dir: &Path,
    llama_src: &Path,
    out_dir: &Path,
    target_os: &TargetOs,
    target_triple: &str,
    android_ndk: Option<&AndroidNdk>,
) {
    let mut builder = create_base_builder(wrapper_dir, llama_src);

    if target_os.is_android()
        && let Some(ndk) = android_ndk
    {
        builder = configure_android_bindgen(builder, ndk, target_triple);
    }

    if target_os.is_msvc() {
        builder = configure_msvc_bindgen(builder, target_triple);
    }

    let bindings = builder
        .generate()
        .expect("bindgen failed to generate FFI bindings");

    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write generated bindings to file");

    debug_log!("Bindings Created");
}

fn create_base_builder(wrapper_dir: &Path, llama_src: &Path) -> bindgen::Builder {
    bindgen::Builder::default()
        .header(wrapper_dir.join("wrapper.h").to_string_lossy())
        .header(wrapper_dir.join("wrapper_mtmd.h").to_string_lossy())
        .clang_arg(format!("-I{}", llama_src.join("include").display()))
        .clang_arg(format!("-I{}", llama_src.join("ggml/include").display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .derive_partialeq(true)
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("gguf_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("llama_rs_.*")
        .allowlist_type("llama_rs_.*")
        .allowlist_function("mtmd_.*")
        .allowlist_type("mtmd_.*")
        .blocklist_function("ggml_fopen")
        .blocklist_function("gguf_init_from_file_ptr")
        .blocklist_function("gguf_write_to_file_ptr")
        .blocklist_function("llama_model_load_from_file_ptr")
        .blocklist_type("FILE")
        .blocklist_type("_IO_.*")
        .blocklist_type("_iobuf")
        .blocklist_type("__BindgenBitfieldUnit")
        .prepend_enum_name(false)
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

    if let Some(ref builtin_includes) = ndk.clang_builtin_includes {
        builder = builder.clang_arg("-isystem").clang_arg(builtin_includes);
    }

    builder = builder
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include/{}", ndk.sysroot, ndk.target_prefix))
        .clang_arg("-isystem")
        .clang_arg(format!("{}/usr/include", ndk.sysroot))
        .clang_arg("-include")
        .clang_arg("stdbool.h")
        .clang_arg("-include")
        .clang_arg("stdint.h");

    if env::var("CARGO_SUBCOMMAND").as_deref() == Ok("ndk") {
        // SAFETY: build scripts are single-threaded, so modifying env is safe.
        unsafe {
            env::set_var(
                "BINDGEN_EXTRA_CLANG_ARGS",
                format!("--target={target_triple}"),
            );
        }
    }

    builder
}

fn apply_msvc_include_paths(
    mut builder: bindgen::Builder,
    include_paths: Option<&str>,
) -> bindgen::Builder {
    let Some(include_paths) = include_paths else {
        return builder;
    };

    for include_path in split_msvc_include_paths(include_paths) {
        builder = builder.clang_arg("-isystem").clang_arg(&include_path);
        debug_log!("Added MSVC include path: {}", include_path);
    }

    builder
}

fn split_msvc_include_paths(include_paths: &str) -> Vec<String> {
    include_paths
        .split(';')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect()
}

fn configure_msvc_bindgen(mut builder: bindgen::Builder, target_triple: &str) -> bindgen::Builder {
    let out_dir_str = env::var("OUT_DIR").unwrap_or_default();
    let dummy_c = Path::new(&out_dir_str).join("dummy.c");

    if std::fs::write(&dummy_c, "int main() { return 0; }").is_err() {
        return builder;
    }

    let mut cc_build = cc::Build::new();
    cc_build.file(&dummy_c);

    let Ok(compiler) = cc_build.try_get_compiler() else {
        return builder;
    };

    let msvc_include_paths = compiler
        .env()
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case("INCLUDE"))
        .map(|(_, value)| value.clone());

    builder = apply_msvc_include_paths(
        builder,
        msvc_include_paths
            .as_ref()
            .map(|paths| paths.to_string_lossy())
            .as_deref(),
    );

    builder = builder
        .clang_arg(format!("--target={target_triple}"))
        .clang_arg("-fms-compatibility")
        .clang_arg("-fms-extensions");

    debug_log!(
        "Configured bindgen with MSVC toolchain for target: {}",
        target_triple
    );

    builder
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::android_ndk::AndroidNdk;
    use crate::cc_test_environment::with_cc_environment;
    use crate::scratch_dir::ScratchDir;

    use super::apply_msvc_include_paths;
    use super::configure_android_bindgen;
    use super::configure_msvc_bindgen;
    use super::create_base_builder;
    use super::split_msvc_include_paths;

    fn android_ndk(clang_builtin_includes: Option<String>) -> AndroidNdk {
        AndroidNdk {
            ndk_path: "/ndk".to_owned(),
            api_level: "28".to_owned(),
            abi: "arm64-v8a",
            host_tag: "darwin-x86_64",
            toolchain_path: "/ndk/toolchains/llvm/prebuilt/darwin-x86_64".to_owned(),
            sysroot: "/ndk/toolchains/llvm/prebuilt/darwin-x86_64/sysroot".to_owned(),
            target_prefix: "aarch64-linux-android",
            clang_builtin_includes,
        }
    }

    fn rendered(builder: &bindgen::Builder) -> String {
        format!("{builder:?}")
    }

    #[test]
    #[serial]
    fn bindings_are_generated_and_respect_the_allowlist() {
        let scratch = ScratchDir::new("bindgen-generate");
        let llama_src = scratch.path().join("llama.cpp");
        std::fs::create_dir_all(llama_src.join("include")).expect("include must be creatable");
        std::fs::create_dir_all(llama_src.join("ggml/include")).expect("ggml must be creatable");
        std::fs::write(
            scratch.path().join("wrapper.h"),
            b"int llama_probe(void);\nint unrelated_probe(void);\n",
        )
        .expect("wrapper header must be writable");
        std::fs::write(
            scratch.path().join("wrapper_mtmd.h"),
            b"int mtmd_probe(void);\n",
        )
        .expect("mtmd header must be writable");

        super::generate_bindings(
            scratch.path(),
            &llama_src,
            scratch.path(),
            &crate::target_os::TargetOs::from_target_triple("aarch64-apple-darwin")
                .expect("supported triple"),
            "aarch64-apple-darwin",
            None,
        );

        let generated = std::fs::read_to_string(scratch.path().join("bindings.rs"))
            .expect("bindings must be written");

        assert!(generated.contains("llama_probe"), "allowlisted llama_*");
        assert!(generated.contains("mtmd_probe"), "allowlisted mtmd_*");
        assert!(
            !generated.contains("unrelated_probe"),
            "symbols outside the allowlist must be excluded"
        );
    }

    #[test]
    fn the_base_builder_allowlists_the_upstream_prefixes() {
        let builder =
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama"));
        let description = rendered(&builder);

        for expected in ["llama_.*", "ggml_.*", "mtmd_.*", "gguf_.*"] {
            assert!(description.contains(expected), "missing {expected}");
        }
        assert!(description.contains("/llama/include"));
        assert!(description.contains("/llama/ggml/include"));
    }

    #[test]
    #[serial]
    fn the_android_configuration_adds_the_sysroot_and_builtin_includes() {
        unsafe { std::env::remove_var("CARGO_SUBCOMMAND") };
        let ndk = android_ndk(Some("/ndk/lib/clang/18/include".to_owned()));

        let builder = configure_android_bindgen(
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
            &ndk,
            "aarch64-linux-android",
        );
        let description = rendered(&builder);

        assert!(description.contains(&format!("--sysroot={}", ndk.sysroot)));
        assert!(description.contains("-D__ANDROID_API__=28"));
        assert!(description.contains("/ndk/lib/clang/18/include"));
        assert!(description.contains("stdbool.h"));
    }

    #[test]
    #[serial]
    fn the_android_configuration_tolerates_absent_builtin_includes() {
        unsafe { std::env::remove_var("CARGO_SUBCOMMAND") };

        let builder = configure_android_bindgen(
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
            &android_ndk(None),
            "aarch64-linux-android",
        );

        assert!(rendered(&builder).contains("-D__ANDROID__"));
    }

    #[test]
    #[serial]
    fn the_cargo_ndk_subcommand_sets_the_extra_clang_target() {
        unsafe {
            std::env::set_var("CARGO_SUBCOMMAND", "ndk");
            std::env::remove_var("BINDGEN_EXTRA_CLANG_ARGS");
        }

        let _ = configure_android_bindgen(
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
            &android_ndk(None),
            "aarch64-linux-android",
        );

        let extra = std::env::var("BINDGEN_EXTRA_CLANG_ARGS").unwrap_or_default();

        unsafe {
            std::env::remove_var("CARGO_SUBCOMMAND");
            std::env::remove_var("BINDGEN_EXTRA_CLANG_ARGS");
        }

        assert_eq!(extra, "--target=aarch64-linux-android");
    }

    #[test]
    #[serial]
    fn the_msvc_configuration_adds_compatibility_flags() {
        let scratch = ScratchDir::new("bindgen-msvc");

        with_cc_environment(scratch.path(), || {
            let builder = configure_msvc_bindgen(
                create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
                "x86_64-pc-windows-msvc",
            );
            let description = rendered(&builder);

            assert!(description.contains("-fms-compatibility"));
            assert!(description.contains("-fms-extensions"));
            assert!(description.contains("--target=x86_64-pc-windows-msvc"));
        });
    }

    #[test]
    #[serial]
    fn an_unwritable_out_dir_leaves_the_msvc_builder_untouched() {
        unsafe { std::env::set_var("OUT_DIR", "/definitely/not/a/directory") };

        let builder = configure_msvc_bindgen(
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
            "x86_64-pc-windows-msvc",
        );

        unsafe { std::env::remove_var("OUT_DIR") };

        assert!(!rendered(&builder).contains("-fms-compatibility"));
    }

    #[test]
    #[serial]
    fn an_msvc_target_takes_the_msvc_configuration_path() {
        let scratch = ScratchDir::new("bindgen-msvc-path");
        let llama_src = scratch.path().join("llama.cpp");
        std::fs::create_dir_all(llama_src.join("ggml/include")).expect("tree must be creatable");
        std::fs::write(
            scratch.path().join("wrapper.h"),
            b"int llama_probe(void);\n",
        )
        .expect("header must be writable");
        std::fs::write(
            scratch.path().join("wrapper_mtmd.h"),
            b"int mtmd_probe(void);\n",
        )
        .expect("header must be writable");

        with_cc_environment(scratch.path(), || {
            unsafe { std::env::set_var("INCLUDE", format!("{};", scratch.path().display())) };

            super::generate_bindings(
                scratch.path(),
                &llama_src,
                scratch.path(),
                &crate::target_os::TargetOs::from_target_triple("x86_64-pc-windows-msvc")
                    .expect("msvc triple"),
                "x86_64-pc-windows-msvc",
                None,
            );

            unsafe { std::env::remove_var("INCLUDE") };
        });

        assert!(
            scratch.path().join("bindings.rs").exists(),
            "bindings must still be produced for an msvc target"
        );
    }

    #[test]
    #[serial]
    fn an_android_target_takes_the_ndk_configuration_path() {
        let scratch = ScratchDir::new("bindgen-android-path");
        let llama_src = scratch.path().join("llama.cpp");
        std::fs::create_dir_all(llama_src.join("ggml/include")).expect("tree must be creatable");
        std::fs::write(
            scratch.path().join("wrapper.h"),
            b"int llama_probe(void);\n",
        )
        .expect("header must be writable");
        std::fs::write(
            scratch.path().join("wrapper_mtmd.h"),
            b"int mtmd_probe(void);\n",
        )
        .expect("header must be writable");
        let sysroot = scratch.path().join("sysroot");
        std::fs::create_dir_all(sysroot.join("usr/include")).expect("sysroot must be creatable");

        let mut ndk = android_ndk(None);
        ndk.sysroot = sysroot.to_string_lossy().into_owned();

        unsafe { std::env::remove_var("CARGO_SUBCOMMAND") };

        super::generate_bindings(
            scratch.path(),
            &llama_src,
            scratch.path(),
            &crate::target_os::TargetOs::from_target_triple("aarch64-linux-android")
                .expect("android triple"),
            "aarch64-linux-android",
            Some(&ndk),
        );

        assert!(
            scratch.path().join("bindings.rs").exists(),
            "bindings must still be produced for an android target"
        );
    }

    #[test]
    #[serial]
    fn a_missing_compiler_environment_leaves_the_msvc_builder_untouched() {
        let scratch = ScratchDir::new("bindgen-nocompiler");

        unsafe {
            std::env::set_var("OUT_DIR", scratch.path());
            std::env::remove_var("HOST");
            std::env::remove_var("TARGET");
            std::env::remove_var("OPT_LEVEL");
        }

        let builder = configure_msvc_bindgen(
            create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama")),
            "x86_64-pc-windows-msvc",
        );

        unsafe { std::env::remove_var("OUT_DIR") };

        assert!(
            !rendered(&builder).contains("-fms-compatibility"),
            "without a resolvable compiler the builder must be returned unchanged"
        );
    }

    #[test]
    fn msvc_include_paths_are_split_and_emptied_entries_dropped() {
        assert_eq!(
            split_msvc_include_paths("C:\\a;C:\\b;;C:\\c"),
            vec!["C:\\a".to_owned(), "C:\\b".to_owned(), "C:\\c".to_owned()]
        );
        assert!(split_msvc_include_paths("").is_empty());
        assert!(split_msvc_include_paths(";;").is_empty());
    }

    #[test]
    fn msvc_include_paths_are_applied_when_the_toolchain_reports_them() {
        let base =
            || create_base_builder(std::path::Path::new("/sys"), std::path::Path::new("/llama"));

        let applied = apply_msvc_include_paths(base(), Some("/win/ucrt;/win/shared"));
        let description = rendered(&applied);

        assert!(description.contains("/win/ucrt"));
        assert!(description.contains("/win/shared"));

        let untouched = rendered(&apply_msvc_include_paths(base(), None));

        assert!(!untouched.contains("/win/ucrt"));
    }
}
