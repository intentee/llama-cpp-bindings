use std::ffi::CStr;
use std::ffi::c_void;
use std::ptr;

/// The log callback ggml currently calls, with the user data it hands to it.
pub struct RegisteredLogCallback {
    callback: llama_cpp_bindings_sys::bindings::ggml_log_callback,
    user_data: *mut c_void,
}

impl RegisteredLogCallback {
    #[must_use]
    pub fn of_ggml() -> Self {
        let mut callback: llama_cpp_bindings_sys::bindings::ggml_log_callback = None;
        let mut user_data: *mut c_void = ptr::null_mut();

        unsafe {
            llama_cpp_bindings_sys::bindings::ggml_log_get(&raw mut callback, &raw mut user_data);
        }

        Self {
            callback,
            user_data,
        }
    }

    /// Hands `text` to the registered callback the way llama.cpp does, and tells whether a
    /// callback was registered to receive it.
    #[must_use]
    pub fn emit(
        &self,
        level: llama_cpp_bindings_sys::bindings::ggml_log_level,
        text: &CStr,
    ) -> bool {
        self.callback
            .map(|callback| unsafe { callback(level, text.as_ptr(), self.user_data) })
            .is_some()
    }
}
