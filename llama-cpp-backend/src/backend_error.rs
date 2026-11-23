use std::str::Utf8Error;

use llama_cpp_path_c_string::path_c_string_error::PathCStringError;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum BackendError {
    #[error("BackendAlreadyInitialized")]
    AlreadyInitialized,
    #[error("the backend directory path cannot be handed to ggml")]
    BackendDirectoryPathInvalid(#[source] PathCStringError),
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
    UnknownDeviceType(llama_cpp_bindings_sys::bindings::ggml_backend_dev_type),
    #[error("NUMA strategy {0} is not a value ggml defines")]
    UnknownNumaStrategy(llama_cpp_bindings_sys::bindings::ggml_numa_strategy),
}
