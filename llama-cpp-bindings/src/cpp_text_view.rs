use std::ffi::c_char;
use std::slice;
use std::str;
use std::str::Utf8Error;

/// Borrows the UTF-8 text of a C++ string llama.cpp exposes as a pointer and a length.
///
/// # Safety
///
/// `text` must point to `length` readable bytes that stay alive and unchanged for `'text`.
///
/// # Errors
///
/// Returns [`Utf8Error`] when the bytes are not UTF-8.
pub const unsafe fn cpp_text_view<'text>(
    text: *const c_char,
    length: usize,
) -> Result<&'text str, Utf8Error> {
    str::from_utf8(unsafe { slice::from_raw_parts(text.cast::<u8>(), length) })
}
