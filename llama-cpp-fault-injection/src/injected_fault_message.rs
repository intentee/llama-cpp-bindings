use std::ffi::CStr;
use std::str::Utf8Error;

/// The message of the C++ exception an armed fault point throws.
///
/// # Errors
///
/// Returns [`Utf8Error`] when the wrappers' fault message is not UTF-8.
pub fn injected_fault_message() -> Result<String, Utf8Error> {
    unsafe { CStr::from_ptr(llama_cpp_bindings_sys::llama_rs_fault_injection_message()) }
        .to_str()
        .map(str::to_owned)
}
