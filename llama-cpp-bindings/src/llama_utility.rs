use std::ffi::{CStr, CString};

use crate::error::{LlamaCppError, Result};

/// Returns true if the given status indicates success.
#[must_use]
pub fn status_is_ok(status: llama_cpp_bindings_sys::llama_rs_status) -> bool {
    status == llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK
}

/// Converts a status code to its underlying `i32` representation.
#[must_use]
pub fn status_to_i32(status: llama_cpp_bindings_sys::llama_rs_status) -> i32 {
    status
}

/// get the time (in microseconds) according to llama.cpp
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

/// get the max number of devices according to llama.cpp (this is generally cuda devices)
/// ```
/// # use llama_cpp_bindings::max_devices;
/// let max_devices = max_devices();
/// assert!(max_devices >= 0);
/// ```
#[must_use]
pub fn max_devices() -> usize {
    unsafe { llama_cpp_bindings_sys::llama_max_devices() }
}

/// is memory mapping supported according to llama.cpp
/// ```
/// # use llama_cpp_bindings::mmap_supported;
/// let mmap_supported = mmap_supported();
/// if mmap_supported {
///   println!("mmap_supported!");
/// }
/// ```
#[must_use]
pub fn mmap_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mmap() }
}

/// is memory locking supported according to llama.cpp
/// ```
/// # use llama_cpp_bindings::mlock_supported;
/// let mlock_supported = mlock_supported();
/// if mlock_supported {
///    println!("mlock_supported!");
/// }
/// ```
#[must_use]
pub fn mlock_supported() -> bool {
    unsafe { llama_cpp_bindings_sys::llama_supports_mlock() }
}

/// Convert a JSON schema string into a llama.cpp grammar string.
///
/// # Errors
/// Returns an error if the schema contains null bytes or the conversion fails.
pub fn json_schema_to_grammar(schema_json: &str) -> Result<String> {
    let schema_cstr = CString::new(schema_json)
        .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
    let mut out = std::ptr::null_mut();
    let rc = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema_cstr.as_ptr(),
            false,
            &raw mut out,
        )
    };

    let result = {
        if !status_is_ok(rc) || out.is_null() {
            return Err(LlamaCppError::JsonSchemaToGrammarError(format!(
                "ffi error {}",
                status_to_i32(rc)
            )));
        }
        let grammar_bytes = unsafe { CStr::from_ptr(out) }.to_bytes().to_vec();
        let grammar = String::from_utf8(grammar_bytes)
            .map_err(|err| LlamaCppError::JsonSchemaToGrammarError(err.to_string()))?;
        Ok(grammar)
    };

    unsafe { llama_cpp_bindings_sys::llama_rs_string_free(out) };

    result
}

/// Get the time in microseconds according to ggml
///
/// ```
/// # use std::time::Duration;
/// # use llama_cpp_bindings::llama_backend::LlamaBackend;
/// let backend = LlamaBackend::init().unwrap();
/// use llama_cpp_bindings::ggml_time_us;
///
/// let start = ggml_time_us();
///
/// std::thread::sleep(Duration::from_micros(10));
///
/// let end = ggml_time_us();
///
/// let elapsed = end - start;
///
/// assert!(elapsed >= 10)
#[must_use]
pub fn ggml_time_us() -> i64 {
    unsafe { llama_cpp_bindings_sys::ggml_time_us() }
}

#[cfg(test)]
mod tests {
    use super::{json_schema_to_grammar, status_is_ok, status_to_i32};

    #[test]
    fn status_is_ok_for_ok_status() {
        assert!(status_is_ok(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK));
    }

    #[test]
    fn status_is_ok_for_error_status() {
        assert!(!status_is_ok(1));
        assert!(!status_is_ok(-1));
    }

    #[test]
    fn status_to_i32_preserves_value() {
        assert_eq!(
            status_to_i32(llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK),
            llama_cpp_bindings_sys::LLAMA_RS_STATUS_OK
        );
        assert_eq!(status_to_i32(42), 42);
        assert_eq!(status_to_i32(-1), -1);
    }

    #[test]
    fn json_schema_to_grammar_simple_object() {
        let schema = r#"{"type": "object", "properties": {"name": {"type": "string"}}}"#;
        let result = json_schema_to_grammar(schema);

        assert!(result.is_ok(), "expected valid grammar, got {result:?}");
        assert!(!result.unwrap().is_empty());
    }

    #[test]
    fn json_schema_to_grammar_with_null_byte_returns_error() {
        let schema = "{\x00}";
        let result = json_schema_to_grammar(schema);

        assert!(result.is_err());
    }

    #[test]
    fn json_schema_to_grammar_simple_string() {
        let schema = r#"{"type": "string"}"#;
        let result = json_schema_to_grammar(schema);

        assert!(result.is_ok(), "expected valid grammar, got {result:?}");
    }

    #[test]
    fn llama_time_us_returns_positive() {
        let time = super::llama_time_us();

        assert!(time > 0);
    }

    #[test]
    fn max_devices_is_non_negative() {
        let devices = super::max_devices();

        assert!(devices > 0);
    }

    #[test]
    fn ggml_time_us_returns_positive() {
        let time = super::ggml_time_us();

        assert!(time > 0);
    }
}
