/// Converts a status code to its underlying `i32` representation.
#[must_use]
pub fn status_to_i32(status: llama_cpp_bindings_sys::llama_rs_status) -> i32 {
    status
}
