use crate::error::unknown_backend_device_type::UnknownBackendDeviceType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlamaBackendDeviceType {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Meta,
}

impl TryFrom<llama_cpp_bindings_sys::ggml_backend_dev_type> for LlamaBackendDeviceType {
    type Error = UnknownBackendDeviceType;

    fn try_from(value: llama_cpp_bindings_sys::ggml_backend_dev_type) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU => Ok(Self::Cpu),
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU => Ok(Self::Gpu),
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU => Ok(Self::IntegratedGpu),
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL => Ok(Self::Accelerator),
            llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_META => Ok(Self::Meta),
            unknown => Err(UnknownBackendDeviceType(unknown)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::error::unknown_backend_device_type::UnknownBackendDeviceType;
    use crate::llama_backend_device_type::LlamaBackendDeviceType;

    #[test]
    fn every_ggml_device_type_converts_to_its_variant() {
        assert_eq!(
            [
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_CPU,
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_GPU,
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_IGPU,
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_ACCEL,
                llama_cpp_bindings_sys::GGML_BACKEND_DEVICE_TYPE_META,
            ]
            .map(LlamaBackendDeviceType::try_from),
            [
                Ok(LlamaBackendDeviceType::Cpu),
                Ok(LlamaBackendDeviceType::Gpu),
                Ok(LlamaBackendDeviceType::IntegratedGpu),
                Ok(LlamaBackendDeviceType::Accelerator),
                Ok(LlamaBackendDeviceType::Meta),
            ]
        );
    }

    #[test]
    fn a_value_ggml_does_not_define_is_refused() {
        assert_eq!(
            LlamaBackendDeviceType::try_from(9999),
            Err(UnknownBackendDeviceType(9999))
        );
    }
}
