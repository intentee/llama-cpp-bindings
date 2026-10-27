use std::ptr;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_input_text_part::MtmdInputTextPart;

pub enum MtmdInputPart<'bitmap> {
    Bitmap(&'bitmap MtmdBitmap),
    Text(MtmdInputTextPart),
}

impl MtmdInputPart<'_> {
    #[must_use]
    pub const fn ffi_part(&self) -> llama_cpp_bindings_sys::mtmd_input_part {
        match self {
            Self::Bitmap(bitmap) => llama_cpp_bindings_sys::mtmd_input_part {
                text: ptr::null(),
                bitmap: bitmap.bitmap.as_ptr().cast_const(),
            },
            Self::Text(text_part) => llama_cpp_bindings_sys::mtmd_input_part {
                text: ptr::from_ref(text_part.input_text()),
                bitmap: ptr::null(),
            },
        }
    }
}
