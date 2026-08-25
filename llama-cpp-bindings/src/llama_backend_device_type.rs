#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaBackendDeviceType {
    Cpu,
    Accelerator,
    Gpu,
    IntegratedGpu,
    Unknown,
}

impl LlamaBackendDeviceType {
    #[must_use]
    pub const fn from_raw(raw_type: llama_cpp_bindings_sys::ggml_backend_dev_type) -> Self {
        match raw_type {
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU => Self::Cpu,
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL => Self::Accelerator,
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU => Self::Gpu,
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU => Self::IntegratedGpu,
            _ => Self::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LlamaBackendDeviceType;

    #[test]
    fn device_type_from_raw_all_variants() {
        assert_eq!(
            LlamaBackendDeviceType::from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU),
            LlamaBackendDeviceType::Cpu
        );
        assert_eq!(
            LlamaBackendDeviceType::from_raw(
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL
            ),
            LlamaBackendDeviceType::Accelerator
        );
        assert_eq!(
            LlamaBackendDeviceType::from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU),
            LlamaBackendDeviceType::Gpu
        );
        assert_eq!(
            LlamaBackendDeviceType::from_raw(llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU),
            LlamaBackendDeviceType::IntegratedGpu
        );
        assert_eq!(
            LlamaBackendDeviceType::from_raw(9999),
            LlamaBackendDeviceType::Unknown
        );
    }
}
