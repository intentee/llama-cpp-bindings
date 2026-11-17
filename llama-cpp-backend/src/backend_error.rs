use std::ffi::NulError;
use std::path::PathBuf;
use std::str::Utf8Error;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BackendError {
    #[error("BackendAlreadyInitialized")]
    AlreadyInitialized,
    #[error("backend directory path is not valid UTF-8: {0}")]
    BackendDirectoryPathNotUtf8(PathBuf),
    #[error("backend directory path contains a null byte")]
    BackendDirectoryPathContainsNullByte(#[source] NulError),
    #[error("the device name cannot be read: ggml reported no text")]
    DeviceNameMissing,
    #[error("the device name cannot be read: ggml reported text that is not valid UTF-8")]
    DeviceNameNotUtf8(#[source] Utf8Error),
    #[error("the device description cannot be read: ggml reported no text")]
    DeviceDescriptionMissing,
    #[error("the device description cannot be read: ggml reported text that is not valid UTF-8")]
    DeviceDescriptionNotUtf8(#[source] Utf8Error),
    #[error("the device's backend name cannot be read: ggml reported no text")]
    DeviceBackendNameMissing,
    #[error("the device's backend name cannot be read: ggml reported text that is not valid UTF-8")]
    DeviceBackendNameNotUtf8(#[source] Utf8Error),
    #[error("backend device type {0} is not a value ggml defines")]
    UnknownDeviceType(llama_cpp_bindings_sys::ggml_backend_dev_type),
    #[error("NUMA strategy {0} is not a value ggml defines")]
    UnknownNumaStrategy(llama_cpp_bindings_sys::ggml_numa_strategy),
}
