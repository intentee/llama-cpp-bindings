use std::ffi::CStr;

/// The message of the C++ exception an armed fault point throws.
///
/// # Panics
///
/// Panics when the wrappers' fault message is not UTF-8.
#[must_use]
pub fn injected_fault_message() -> String {
    unsafe { CStr::from_ptr(llama_cpp_bindings_sys::llama_rs_fault_injection_message()) }
        .to_str()
        .expect("the wrappers' fault message must be UTF-8")
        .to_owned()
}
