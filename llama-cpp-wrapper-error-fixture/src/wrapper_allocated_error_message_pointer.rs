use std::ffi::c_char;
use std::ptr::null_mut;

/// Hands back a message string the C++ wrappers allocated themselves.
///
/// Tests use it to drive the paths that take ownership of such a pointer. The caller
/// owns the allocation and must release it through `read_and_free_cpp_string`.
///
/// # Panics
///
/// Panics when llama.cpp stops rejecting the malformed schema this fixture relies on,
/// or accepts it without storing a message, because the fixture can no longer produce
/// the pointer it promises.
#[must_use]
pub fn wrapper_allocated_error_message_pointer() -> *mut c_char {
    let schema = c"not valid json at all";
    let mut out_grammar: *mut c_char = null_mut();
    let mut out_error: *mut c_char = null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_json_schema_to_grammar(
            schema.as_ptr(),
            false,
            &raw mut out_grammar,
            &raw mut out_error,
        )
    };

    assert_eq!(
        status,
        llama_cpp_bindings_sys::LLAMA_RS_JSON_SCHEMA_TO_GRAMMAR_LLAMA_CPP_THREW_CXX_EXCEPTION,
        "the fixture depends on the llama.cpp json parser rejecting this schema"
    );
    assert!(
        !out_error.is_null(),
        "the wrapper must store a message alongside the exception status"
    );

    out_error
}
