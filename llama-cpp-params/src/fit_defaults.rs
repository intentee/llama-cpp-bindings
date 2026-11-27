use llama_cpp_backend::max_devices::max_devices;
use llama_cpp_bindings_sys::bindings::llama_rs_fit_defaults_values;

/// The memory margin and the smallest context llama.cpp's own tools fit parameters with.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FitDefaults {
    pub margin_per_device: usize,
    pub n_ctx_min: u32,
}

impl FitDefaults {
    #[must_use]
    pub fn of_llama_cpp() -> Self {
        let llama_rs_fit_defaults_values {
            margin_per_device,
            n_ctx_min,
        } = unsafe { llama_cpp_bindings_sys::bindings::llama_rs_fit_defaults() };

        Self {
            margin_per_device,
            n_ctx_min,
        }
    }

    /// The memory margin of every device, the way `LlamaModelParams::fit_params` takes margins.
    #[must_use]
    pub fn margins(&self) -> Vec<usize> {
        vec![self.margin_per_device; max_devices()]
    }
}
