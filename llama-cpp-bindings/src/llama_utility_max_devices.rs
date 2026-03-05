/// Get the max number of devices according to llama.cpp (this is generally cuda devices).
///
/// ```
/// # use llama_cpp_bindings::max_devices;
/// let max_devices = max_devices();
/// assert!(max_devices >= 0);
/// ```
#[must_use]
pub fn max_devices() -> usize {
    unsafe { llama_cpp_bindings_sys::llama_max_devices() }
}
