use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

use crate::backend_error::BackendError;
use crate::numa_strategy::NumaStrategy;

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

impl LlamaBackend {
    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init() -> Result<Self, BackendError> {
        Self::mark_init()?;
        unsafe { llama_cpp_bindings_sys::llama_backend_init() }
        Ok(Self {})
    }

    /// Initializes the backend, then llama.cpp's NUMA support, which needs the CPU backend the
    /// backend initialization loads.
    ///
    /// # Errors
    /// Returns an error if the backend was already initialized.
    pub fn init_numa(strategy: NumaStrategy) -> Result<Self, BackendError> {
        let backend = Self::init()?;

        unsafe {
            llama_cpp_bindings_sys::llama_numa_init(
                llama_cpp_bindings_sys::ggml_numa_strategy::from(strategy),
            );
        }

        Ok(backend)
    }

    #[must_use]
    pub fn supports_gpu_offload() -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_gpu_offload() }
    }

    #[must_use]
    pub fn supports_mlock() -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
    }

    #[must_use]
    pub fn supports_mmap() -> bool {
        unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
    }

    fn mark_init() -> Result<(), BackendError> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_was_uninitialized) => Ok(()),
            Err(_was_already_initialized) => Err(BackendError::AlreadyInitialized),
        }
    }
}

impl Drop for LlamaBackend {
    fn drop(&mut self) {
        LLAMA_BACKEND_INITIALIZED.store(false, SeqCst);
        unsafe { llama_cpp_bindings_sys::llama_backend_free() }
    }
}
