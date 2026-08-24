mod android_ndk;
mod bindgen_config;
mod cmake_config;
mod cpp_wrapper;
mod library_linking;
mod native_sources;
mod rebuild_tracking;
mod target_os;

use std::env;
use std::path::{Path, PathBuf};

use android_ndk::AndroidNdk;
use target_os::TargetOs;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("environment variable {name} could not be read: {source}")]
    Environment {
        name: &'static str,
        #[source]
        source: env::VarError,
    },
    #[error("{0}")]
    Target(String),
    #[error(transparent)]
    AndroidNdk(#[from] android_ndk::AndroidNdkDetectionError),
    #[error("bindgen failed: {0}")]
    Bindgen(#[source] bindgen::BindgenError),
    #[error("generated bindings could not be written: {0}")]
    BindingsWrite(#[source] std::io::Error),
    #[error("native compiler setup failed: {0}")]
    NativeCompiler(#[source] cc::Error),
    #[error("native wrapper compilation failed: {0}")]
    NativeWrapper(#[source] cc::Error),
    #[error("filesystem operation failed for {path}: {source}")]
    Filesystem {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("environment path {name} does not name a directory: {path}")]
    EnvironmentDirectory { name: &'static str, path: PathBuf },
}

fn required_env(name: &'static str) -> Result<String, BuildError> {
    env::var(name).map_err(|source| BuildError::Environment { name, source })
}

fn optional_env(name: &'static str) -> Result<Option<String>, BuildError> {
    match env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(source) => Err(BuildError::Environment { name, source }),
    }
}

#[macro_export]
macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("BUILD_DEBUG").is_ok() {
            println!("cargo:warning=[DEBUG] {}", format!($($arg)*));
        }
    };
}

#[derive(Debug)]
pub struct BuildContext {
    pub out_dir: PathBuf,
    pub cmake_dir: PathBuf,
    pub llama_src: PathBuf,
    pub target_os: TargetOs,
    pub target_triple: String,
    pub build_shared_libs: bool,
    pub profile: String,
    pub static_crt: bool,
    pub android_ndk: Option<AndroidNdk>,
}

impl BuildContext {
    fn detect() -> Result<Self, BuildError> {
        let target_triple = required_env("TARGET")?;
        let target_os = TargetOs::from_target_triple(&target_triple).map_err(BuildError::Target)?;
        let out_dir = PathBuf::from(required_env("OUT_DIR")?);
        let manifest_dir = required_env("CARGO_MANIFEST_DIR")?;
        let llama_src = Path::new(&manifest_dir).join("llama.cpp");

        let build_shared_libs = cfg!(feature = "dynamic-link");
        let profile = native_profile(&required_env("PROFILE")?);
        let static_crt = optional_env("CARGO_CFG_TARGET_FEATURE")?
            .unwrap_or_default()
            .split(',')
            .any(|feature| feature == "crt-static");

        let android_ndk = if target_os.is_android() {
            Some(AndroidNdk::detect(&target_triple)?)
        } else {
            None
        };

        let cmake_dir = out_dir.join("cmake");

        debug_log!("TARGET: {}", target_triple);
        debug_log!("CARGO_MANIFEST_DIR: {}", manifest_dir);
        debug_log!("OUT_DIR: {}", out_dir.display());
        debug_log!("CMAKE_DIR: {}", cmake_dir.display());
        debug_log!("BUILD_SHARED: {}", build_shared_libs);

        Ok(Self {
            out_dir,
            cmake_dir,
            llama_src,
            target_os,
            target_triple,
            build_shared_libs,
            profile,
            static_crt,
            android_ndk,
        })
    }
}

fn native_profile(cargo_profile: &str) -> String {
    match cargo_profile {
        "debug" => "Release".to_owned(),
        "release" => "Release".to_owned(),
        other => other.to_owned(),
    }
}

pub fn build() -> Result<(), BuildError> {
    let context = BuildContext::detect()?;

    rebuild_tracking::register_rebuild_triggers(&context.llama_src);

    bindgen_config::generate_bindings(
        &context.llama_src,
        &context.out_dir,
        &context.target_os,
        &context.target_triple,
        context.android_ndk.as_ref(),
    )?;

    cpp_wrapper::compile_cpp_wrappers(&context.llama_src, &context.target_os)?;

    let build_dir = cmake_config::configure_and_build(&context)?;

    library_linking::link_libraries(
        &context.cmake_dir,
        &build_dir,
        &context.target_os,
        &context.target_triple,
        context.build_shared_libs,
        &context.profile,
    )?;

    Ok(())
}

#[cfg(test)]
mod build_context_value_tests {
    use super::native_profile;

    #[test]
    fn cargo_debug_and_release_profiles_both_use_optimized_native_code() {
        assert_eq!(native_profile("debug"), "Release");
        assert_eq!(native_profile("release"), "Release");
    }

    #[test]
    fn custom_cargo_profile_name_is_preserved_for_cmake() {
        assert_eq!(native_profile("RelWithDebInfo"), "RelWithDebInfo");
    }
}
