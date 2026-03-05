/// Get the time (in microseconds) according to llama.cpp.
///
/// ```
/// # use llama_cpp_bindings::llama_time_us;
/// # use llama_cpp_bindings::llama_backend::LlamaBackend;
/// let backend = LlamaBackend::init().unwrap();
/// let time = llama_time_us();
/// assert!(time > 0);
/// ```
#[must_use]
pub fn llama_time_us() -> i64 {
    unsafe { llama_cpp_bindings_sys::llama_time_us() }
}
