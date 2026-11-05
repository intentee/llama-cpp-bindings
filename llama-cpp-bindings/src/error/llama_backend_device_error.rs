use crate::error::backend_device_text_error::BackendDeviceTextError;
use crate::error::unknown_backend_device_type::UnknownBackendDeviceType;

#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaBackendDeviceError {
    #[error("the device name cannot be read: {0}")]
    Name(#[source] BackendDeviceTextError),
    #[error("the device description cannot be read: {0}")]
    Description(#[source] BackendDeviceTextError),
    #[error("the device's backend name cannot be read: {0}")]
    BackendName(#[source] BackendDeviceTextError),
    #[error(transparent)]
    UnknownDeviceType(#[from] UnknownBackendDeviceType),
}
