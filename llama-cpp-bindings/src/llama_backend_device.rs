use std::ffi::CStr;
use std::ffi::c_char;
use std::mem;

use crate::error::backend_device_text_error::BackendDeviceTextError;
use crate::error::llama_backend_device_error::LlamaBackendDeviceError;
use crate::llama_backend_device_type::LlamaBackendDeviceType;

fn text_of(text: *const c_char) -> Result<String, BackendDeviceTextError> {
    if text.is_null() {
        return Err(BackendDeviceTextError::Null);
    }

    Ok(unsafe { CStr::from_ptr(text) }.to_str()?.to_owned())
}

fn device_text(text: *const c_char) -> Result<String, BackendDeviceTextError> {
    text_of(unsafe { llama_cpp_bindings_sys::llama_rs_backend_device_text(text) })
}

#[derive(Debug, Clone)]
pub struct LlamaBackendDevice {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub backend: String,
    pub memory_total: usize,
    pub memory_free: usize,
    pub device_type: LlamaBackendDeviceType,
}

impl LlamaBackendDevice {
    fn at_index(device_index: usize) -> Result<Self, LlamaBackendDeviceError> {
        let dev = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(device_index) };
        let props = unsafe {
            let mut props = mem::zeroed();
            llama_cpp_bindings_sys::ggml_backend_dev_get_props(dev, &raw mut props);
            props
        };
        let backend_reg = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_backend_reg(dev) };

        Ok(Self {
            index: device_index,
            name: device_text(props.name).map_err(LlamaBackendDeviceError::Name)?,
            description: device_text(props.description)
                .map_err(LlamaBackendDeviceError::Description)?,
            backend: device_text(unsafe {
                llama_cpp_bindings_sys::ggml_backend_reg_name(backend_reg)
            })
            .map_err(LlamaBackendDeviceError::BackendName)?,
            memory_total: props.memory_total,
            memory_free: props.memory_free,
            device_type: LlamaBackendDeviceType::try_from(unsafe {
                llama_cpp_bindings_sys::llama_rs_backend_device_type(dev)
            })?,
        })
    }

    /// Every device ggml registered, in registration order.
    ///
    /// # Errors
    ///
    /// Returns [`LlamaBackendDeviceError`] when ggml reports a device text that is missing or not
    /// UTF-8, or a device type it does not define.
    pub fn list_all() -> Result<Vec<Self>, LlamaBackendDeviceError> {
        (0..unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() })
            .map(Self::at_index)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::text_of;
    use crate::error::backend_device_text_error::BackendDeviceTextError;
    use crate::llama_backend_device::LlamaBackendDevice;

    #[test]
    fn missing_device_text_is_refused() {
        assert_eq!(text_of(ptr::null()), Err(BackendDeviceTextError::Null));
    }

    #[test]
    fn device_text_that_is_not_utf8_is_refused() {
        let not_utf8 = c"\xff";

        assert_eq!(
            text_of(not_utf8.as_ptr()),
            Err(BackendDeviceTextError::NotUtf8(
                not_utf8.to_str().expect_err("0xff is not UTF-8")
            ))
        );
    }

    #[test]
    fn device_text_is_copied() {
        assert_eq!(text_of(c"CUDA0".as_ptr()), Ok("CUDA0".to_owned()));
    }

    #[test]
    fn every_registered_device_is_listed_at_its_index() {
        let devices = LlamaBackendDevice::list_all().unwrap();

        assert_eq!(
            devices
                .iter()
                .map(|device| device.index)
                .collect::<Vec<_>>(),
            (0..devices.len()).collect::<Vec<_>>()
        );
    }
}
