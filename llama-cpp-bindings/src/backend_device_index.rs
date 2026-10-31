/// The index of `device` among the backend devices ggml has registered.
#[must_use]
pub fn backend_device_index(device: llama_cpp_bindings_sys::ggml_backend_dev_t) -> Option<usize> {
    let device_count = unsafe { llama_cpp_bindings_sys::ggml_backend_dev_count() };

    (0..device_count).position(|device_index| {
        let registered_device =
            unsafe { llama_cpp_bindings_sys::ggml_backend_dev_get(device_index) };

        registered_device == device
    })
}
