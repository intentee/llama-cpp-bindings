use std::env::VarError;
use std::io;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("environment variable {name} could not be read: {source}")]
    Environment {
        name: &'static str,
        #[source]
        source: VarError,
    },
    #[error("unsupported target operating system: {cargo_cfg_target_os}")]
    UnsupportedTargetOs { cargo_cfg_target_os: String },
    #[error("Android NDK not found for target {target_triple}. Set ANDROID_NDK_HOME.")]
    AndroidNdkRootNotConfigured {
        target_triple: String,
        #[source]
        source: VarError,
    },
    #[error("Android NDK path does not exist: {path}")]
    AndroidNdkRootMissing { path: PathBuf },
    #[error("Android NDK toolchain file not found: {path}")]
    AndroidNdkToolchainFileMissing { path: PathBuf },
    #[error("Android NDK toolchain not found at: {path}")]
    AndroidNdkToolchainDirectoryMissing { path: PathBuf },
    #[error("Unsupported host platform for Android NDK")]
    AndroidNdkUnsupportedHostPlatform,
    #[error("Unsupported Android target architecture: {cargo_cfg_target_arch}")]
    UnsupportedAndroidTarget { cargo_cfg_target_arch: String },
    #[error("ANDROID_PLATFORM is set but could not be read: {source}")]
    AndroidPlatformUnreadable {
        #[source]
        source: VarError,
    },
    #[error("Android NDK Clang directory could not be listed at {path}: {source}")]
    AndroidNdkClangDirectoryUnreadable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("Android NDK Clang version directory was not found at {path}")]
    AndroidNdkClangVersionDirectoryMissing { path: PathBuf },
    #[error(
        "Android NDK Clang directory {path} holds more than one version directory: {candidates:?}"
    )]
    AndroidNdkClangVersionDirectoryAmbiguous {
        path: PathBuf,
        candidates: Vec<PathBuf>,
    },
    #[error("Android NDK Clang built-in include directory was not found at {path}")]
    AndroidNdkClangBuiltinIncludesMissing { path: PathBuf },
    #[error("bindgen failed: {0}")]
    Bindgen(#[source] bindgen::BindgenError),
    #[error(
        "bindgen never saw field {type_name}::{field_name}, so it can no longer be kept private"
    )]
    PrivatizedFieldMissing {
        type_name: &'static str,
        field_name: &'static str,
    },
    #[error("generated bindings could not be written: {0}")]
    BindingsWrite(#[source] io::Error),
    #[error("native wrapper compilation failed: {0}")]
    NativeWrapper(#[source] cc::Error),
    #[error("filesystem operation failed for {path}: {source}")]
    Filesystem {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("environment path {name} does not name a directory: {path}")]
    EnvironmentDirectory { name: &'static str, path: PathBuf },
    #[error("cargo directives could not be written: {0}")]
    DirectiveWrite(#[source] io::Error),
    #[error("the CMake file API reply directory {path} could not be listed: {source}")]
    CmakeFileApiReplyDirectoryUnreadable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("CMake wrote no file API reply index into {path}")]
    CmakeFileApiReplyIndexMissing { path: PathBuf },
    #[error("the CMake file API reply file {path} could not be read: {source}")]
    CmakeFileApiReplyFileUnreadable {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("the CMake file API reply file {path} is not the reply this crate queried: {source}")]
    CmakeFileApiReplyFileInvalid {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("the CMake cache has no entry {name}")]
    CmakeCacheEntryMissing { name: String },
    #[error("the CMake cache entry {name} is {value}, neither ON nor OFF")]
    CmakeCacheEntryNotASwitch { name: String, value: String },
    #[error(
        "ggml needs the OpenMP runtime {libraries:?}, which Apple's linker cannot link by its \
         exact file name"
    )]
    OpenmpRuntimeUnlinkableOnApple { libraries: Vec<PathBuf> },
    #[error("the OpenMP runtime library {library} names no file")]
    OpenmpRuntimeLibraryWithoutFileName { library: PathBuf },
}
