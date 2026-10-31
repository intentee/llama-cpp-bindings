use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

use crate::error::Result;
use crate::error::llama_cpp_error::LlamaCppError;
use crate::llama_backend_numa_strategy::NumaStrategy;

#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    fn mark_init() -> Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_was_uninitialized) => Ok(()),
            Err(_was_already_initialized) => Err(LlamaCppError::BackendAlreadyInitialized),
        }
    }

    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init() -> Result<Self> {
        Self::mark_init()?;
        unsafe { llama_cpp_bindings_sys::llama_backend_init() }
        Ok(Self {})
    }

    /// Initializes the backend, then llama.cpp's NUMA support, which needs the CPU backend the
    /// backend initialization loads.
    ///
    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init_numa(strategy: NumaStrategy) -> Result<Self> {
        let backend = Self::init()?;

        unsafe {
            llama_cpp_bindings_sys::llama_numa_init(
                llama_cpp_bindings_sys::ggml_numa_strategy::from(strategy),
            );
        }

        Ok(backend)
    }

    #[must_use]
    pub fn supports_gpu_offload(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_gpu_offload() }
    }

    #[must_use]
    pub fn supports_mmap(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
    }

    #[must_use]
    pub fn supports_mlock(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        LLAMA_BACKEND_INITIALIZED.store(false, SeqCst);
        unsafe { llama_cpp_bindings_sys::llama_backend_free() }
    }
}

#[cfg(test)]
mod tests {
    use std::mem::discriminant;

    use serial_test::serial;

    use super::LlamaBackend;
    use crate::error::llama_cpp_error::LlamaCppError;
    use crate::llama_backend_device::LlamaBackendDevice;
    use crate::llama_backend_device_type::LlamaBackendDeviceType;
    use crate::llama_backend_numa_strategy::NumaStrategy;

    #[test]
    #[serial]
    fn initializing_the_backend_returns_it() {
        assert_eq!(LlamaBackend::init(), Ok(LlamaBackend {}));
    }

    #[test]
    #[serial]
    fn double_init_returns_error() {
        let _backend = LlamaBackend::init().unwrap();
        let second_err = LlamaBackend::init().unwrap_err();

        assert_eq!(
            discriminant(&second_err),
            discriminant(&LlamaCppError::BackendAlreadyInitialized),
            "expected BackendAlreadyInitialized, got {second_err:?}"
        );
    }

    #[test]
    #[serial]
    fn init_numa_returns_error_when_backend_already_initialized() {
        let _backend = LlamaBackend::init().unwrap();
        let second_err = LlamaBackend::init_numa(NumaStrategy::Disabled).unwrap_err();

        assert_eq!(
            discriminant(&second_err),
            discriminant(&LlamaCppError::BackendAlreadyInitialized),
            "expected BackendAlreadyInitialized, got {second_err:?}"
        );
    }

    #[test]
    #[serial]
    fn the_backend_supports_mmap_and_mlock() {
        let backend = LlamaBackend::init().unwrap();

        assert_eq!(
            [backend.supports_mmap(), backend.supports_mlock()],
            [true, true]
        );
    }

    #[test]
    #[serial]
    fn the_backend_offloads_to_a_gpu_exactly_when_one_is_registered() {
        let backend = LlamaBackend::init().unwrap();
        let offloads_to_a_gpu = backend.supports_gpu_offload();

        assert_eq!(
            offloads_to_a_gpu,
            LlamaBackendDevice::list_all().iter().any(|device| matches!(
                device.device_type,
                LlamaBackendDeviceType::Gpu | LlamaBackendDeviceType::IntegratedGpu
            ))
        );
    }

    #[test]
    #[serial]
    fn dropping_backend_allows_reinitialization() {
        let backend = LlamaBackend::init().unwrap();
        drop(backend);
        let backend = LlamaBackend::init();
        assert!(backend.is_ok());
    }

    #[test]
    #[serial]
    fn init_numa_with_a_numa_strategy_initializes_the_backend_numa_support_needs() {
        assert_eq!(
            LlamaBackend::init_numa(NumaStrategy::Distribute),
            Ok(LlamaBackend {})
        );
    }
}
