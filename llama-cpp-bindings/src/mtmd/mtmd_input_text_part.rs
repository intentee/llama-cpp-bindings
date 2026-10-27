use std::ffi::CString;
use std::ffi::NulError;

pub struct MtmdInputTextPart {
    input_text: llama_cpp_bindings_sys::mtmd_input_text,
}

impl MtmdInputTextPart {
    /// # Errors
    ///
    /// Returns [`NulError`] when `text` contains a NUL byte.
    pub fn new(text: &str, parse_special: bool) -> Result<Self, NulError> {
        let text = CString::new(text)?;
        let text_len = text.as_bytes().len();

        Ok(Self {
            input_text: llama_cpp_bindings_sys::mtmd_input_text {
                text: text.into_raw(),
                text_len,
                add_special: false,
                parse_special,
            },
        })
    }

    #[must_use]
    pub const fn input_text(&self) -> &llama_cpp_bindings_sys::mtmd_input_text {
        &self.input_text
    }
}

impl Drop for MtmdInputTextPart {
    fn drop(&mut self) {
        drop(unsafe { CString::from_raw(self.input_text.text.cast_mut()) });
    }
}

#[cfg(test)]
mod tests {
    use super::MtmdInputTextPart;

    #[test]
    fn a_text_part_containing_a_nul_byte_is_rejected() {
        assert_eq!(
            MtmdInputTextPart::new("before\0after", true)
                .err()
                .map(|nul_error| nul_error.nul_position()),
            Some(6)
        );
    }
}
