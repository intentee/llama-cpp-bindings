use std::ffi::{CStr, CString, c_char};
use std::mem;
use std::ptr::{self, NonNull};
use std::slice;

use crate::{ChatParseError, status_is_ok, status_to_i32};

/// Streaming OpenAI-compatible parser state.
#[derive(Debug)]
pub struct ChatParseStateOaicompat {
    /// Raw pointer to the underlying FFI parser state.
    pub state: NonNull<llama_cpp_bindings_sys::llama_rs_chat_parse_state_oaicompat>,
}

impl ChatParseStateOaicompat {
    /// Update the parser with additional text and return OpenAI-compatible deltas as JSON strings.
    ///
    /// # Errors
    /// Returns an error if the FFI call fails or the result is null.
    pub fn update(
        &mut self,
        text_added: &str,
        is_partial: bool,
    ) -> Result<Vec<String>, ChatParseError> {
        let text_cstr = CString::new(text_added)?;
        let mut out_msg: llama_cpp_bindings_sys::llama_rs_chat_msg_oaicompat =
            unsafe { mem::zeroed() };
        let mut out_diffs: *mut llama_cpp_bindings_sys::llama_rs_chat_msg_diff_oaicompat =
            ptr::null_mut();
        let mut out_diffs_count: usize = 0;
        let rc = unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_update_oaicompat(
                self.state.as_ptr(),
                text_cstr.as_ptr(),
                is_partial,
                &raw mut out_msg,
                &raw mut out_diffs,
                &raw mut out_diffs_count,
            )
        };

        let result = {
            if !status_is_ok(rc) {
                return Err(ChatParseError::FfiError(status_to_i32(rc)));
            }
            if out_diffs_count > 0 && out_diffs.is_null() {
                return Err(ChatParseError::NullResult);
            }
            let diffs = if out_diffs_count == 0 {
                &[]
            } else {
                unsafe { slice::from_raw_parts(out_diffs, out_diffs_count) }
            };
            let mut deltas = Vec::with_capacity(diffs.len());

            for diff in diffs {
                let mut out_json: *mut c_char = ptr::null_mut();
                let rc = unsafe {
                    llama_cpp_bindings_sys::llama_rs_chat_msg_diff_to_oaicompat_json(
                        diff,
                        &raw mut out_json,
                    )
                };
                if !status_is_ok(rc) {
                    if !out_json.is_null() {
                        unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };
                    }

                    return Err(ChatParseError::FfiError(status_to_i32(rc)));
                }
                if out_json.is_null() {
                    return Err(ChatParseError::NullResult);
                }
                let bytes = unsafe { CStr::from_ptr(out_json) }.to_bytes().to_vec();
                unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out_json) };
                deltas.push(String::from_utf8(bytes)?);
            }

            Ok(deltas)
        };

        unsafe { llama_cpp_bindings_sys::llama_rs_chat_msg_free_oaicompat(&raw mut out_msg) };
        unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_msg_diff_free_oaicompat(
                out_diffs,
                out_diffs_count,
            );
        };

        result
    }
}

impl Drop for ChatParseStateOaicompat {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_bindings_sys::llama_rs_chat_parse_state_free_oaicompat(self.state.as_ptr())
        };
    }
}
