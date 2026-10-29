use std::ffi::CString;
use std::ffi::NulError;
use std::ptr;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_input_part::MtmdInputPart;

pub enum MtmdFfiInputPart<'bitmap> {
    Bitmap(&'bitmap MtmdBitmap),
    Text(llama_cpp_bindings_sys::mtmd_input_text),
}

impl MtmdFfiInputPart<'_> {
    #[must_use]
    pub const fn ffi_part(&self) -> llama_cpp_bindings_sys::mtmd_input_part {
        match self {
            Self::Bitmap(bitmap) => llama_cpp_bindings_sys::mtmd_input_part {
                text: ptr::null(),
                bitmap: bitmap.bitmap.as_ptr().cast_const(),
            },
            Self::Text(input_text) => llama_cpp_bindings_sys::mtmd_input_part {
                text: ptr::from_ref(input_text),
                bitmap: ptr::null(),
            },
        }
    }
}

impl<'bitmap> TryFrom<&MtmdInputPart<'bitmap>> for MtmdFfiInputPart<'bitmap> {
    type Error = NulError;

    fn try_from(input_part: &MtmdInputPart<'bitmap>) -> Result<Self, Self::Error> {
        match input_part {
            MtmdInputPart::Bitmap(bitmap) => Ok(Self::Bitmap(bitmap)),
            MtmdInputPart::Text(text_part) => {
                let text = CString::new(text_part.text.as_str())?;
                let text_len = text.as_bytes().len();

                Ok(Self::Text(llama_cpp_bindings_sys::mtmd_input_text {
                    text: text.into_raw(),
                    text_len,
                    add_special: false,
                    parse_special: text_part.parse_special,
                }))
            }
        }
    }
}

impl Drop for MtmdFfiInputPart<'_> {
    fn drop(&mut self) {
        if let Self::Text(input_text) = self {
            drop(unsafe { CString::from_raw(input_text.text.cast_mut()) });
        }
    }
}
