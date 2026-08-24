use std::ffi::{CStr, CString, c_char};
use std::path::PathBuf;
use std::ptr::NonNull;
use std::slice;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use super::mtmd_bitmap_error::MtmdBitmapError;
use super::mtmd_context::MtmdContext;

fn cstr_ptr_to_optional_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        let id = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned();

        Some(id)
    }
}

/// # Safety
///
/// `out_bitmap` must be either null or a valid pointer to an `mtmd_bitmap`
/// allocated by `llama_rs_mtmd_bitmap_init_from_file`. `out_error` must be
/// either null or a valid pointer to a null-terminated C string allocated by
/// `llama_rs_dup_string`.
unsafe fn from_file_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_bitmap_init_from_file_status,
    out_bitmap: *mut llama_cpp_bindings_sys::mtmd_bitmap,
    out_error: *mut c_char,
    path: &str,
) -> Result<MtmdBitmap, MtmdBitmapError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK => {
            let bitmap = NonNull::new(out_bitmap).ok_or_else(|| {
                MtmdBitmapError::from(crate::FfiContractError {
                    operation: "llama_rs_mtmd_bitmap_init_from_file",
                    detail: "success status contained a null bitmap",
                })
            })?;
            Ok(MtmdBitmap { bitmap })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_RETURNED_NULL => {
            Err(MtmdBitmapError::FileUnreadable {
                path: PathBuf::from(path),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdBitmapError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_OUT_OF_MEMORY => {
            Err(MtmdBitmapError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_string(out_error, "llama_rs_mtmd_bitmap_init_from_file", "reported a thrown C++ exception without an error message") }?;
            Err(MtmdBitmapError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_bitmap_init_from_file",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_FNAME_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_bitmap_init_from_file",
            detail: "was given a null fname argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_OUT_BITMAP_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_bitmap_init_from_file",
            detail: "was given a null out_bitmap argument",
        }
        .into()),
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_mtmd_bitmap_init_from_file",
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Debug, Clone)]
pub struct MtmdBitmap {
    pub bitmap: NonNull<llama_cpp_bindings_sys::mtmd_bitmap>,
}

const RGB_BYTES_PER_PIXEL: usize = 3;

unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// # Errors
    ///
    /// * `ImageDimensionsTooSmall` - `nx` or `ny` is below the 2x2 minimum
    /// * `ImageDimensionsOverflow` - `nx * ny * 3` does not fit into a buffer length
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `BitmapDecodeFailed` - Underlying C function returned null
    ///
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        if nx < 2 || ny < 2 {
            return Err(MtmdBitmapError::ImageDimensionsTooSmall(nx, ny));
        }

        let expected_len = usize::try_from(nx)
            .ok()
            .and_then(|width| width.checked_mul(usize::try_from(ny).ok()?))
            .and_then(|pixels| pixels.checked_mul(RGB_BYTES_PER_PIXEL))
            .ok_or(MtmdBitmapError::ImageDimensionsOverflow { nx, ny })?;

        if data.len() != expected_len {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let bitmap = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_init(nx, ny, data.as_ptr()) };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::BitmapDecodeFailed)?;

        Ok(Self { bitmap })
    }

    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    ///
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        let bitmap = unsafe {
            llama_cpp_bindings_sys::mtmd_bitmap_init_from_audio(data.len(), data.as_ptr())
        };

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::BitmapDecodeFailed)?;

        Ok(Self { bitmap })
    }

    /// # Errors
    ///
    /// Returns an [`MtmdBitmapError`] variant matching the wrapper's status code.
    pub fn from_file(ctx: &MtmdContext, path: &str) -> Result<Self, MtmdBitmapError> {
        let path_cstr = CString::new(path)?;
        let mut out_bitmap: *mut llama_cpp_bindings_sys::mtmd_bitmap = std::ptr::null_mut();
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
                &raw mut out_bitmap,
                &raw mut out_error,
            )
        };

        unsafe { from_file_status_to_result(status, out_bitmap, out_error, path) }
    }

    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    pub fn from_buffer(ctx: &MtmdContext, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        let bitmap_wrapper = unsafe {
            llama_cpp_bindings_sys::mtmd_helper_bitmap_init_from_buf(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                false,
            )
        };

        let bitmap =
            NonNull::new(bitmap_wrapper.bitmap).ok_or(MtmdBitmapError::BitmapDecodeFailed)?;

        Ok(Self { bitmap })
    }

    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    #[must_use]
    pub fn data(&self) -> &[u8] {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        unsafe { slice::from_raw_parts(ptr, len) }
    }

    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };

        cstr_ptr_to_optional_string(ptr)
    }

    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    pub fn set_id(&self, id: &str) -> Result<(), std::ffi::NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_bindings_sys::mtmd_bitmap_set_id(self.bitmap.as_ptr(), id_cstr.as_ptr());
        }

        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::MtmdBitmap;
    use super::MtmdBitmapError;

    #[test]
    fn cstr_ptr_to_optional_string_returns_none_for_null() {
        assert!(super::cstr_ptr_to_optional_string(std::ptr::null()).is_none());
    }

    #[test]
    fn cstr_ptr_to_optional_string_returns_some_for_valid() {
        let cstr = std::ffi::CString::new("hello").unwrap();
        let result = super::cstr_ptr_to_optional_string(cstr.as_ptr());

        assert_eq!(result, Some("hello".to_string()));
    }

    #[test]
    fn from_image_data_creates_valid_bitmap() {
        let red_pixel: [u8; 3] = [255, 0, 0];
        let image_data: Vec<u8> = red_pixel.repeat(4);
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        assert_eq!(bitmap.nx(), 2);
        assert_eq!(bitmap.ny(), 2);
        assert!(!bitmap.is_audio());
    }

    #[test]
    fn invalid_data_size_returns_error() {
        let too_short = vec![0u8; 5];

        assert_eq!(
            MtmdBitmap::from_image_data(2, 2, &too_short).unwrap_err(),
            MtmdBitmapError::InvalidDataSize
        );
    }

    #[test]
    fn dimensions_whose_rgb_buffer_size_overflows_are_rejected() {
        assert_eq!(
            MtmdBitmap::from_image_data(u32::MAX, u32::MAX, &[0u8; 12]).unwrap_err(),
            MtmdBitmapError::ImageDimensionsOverflow {
                nx: u32::MAX,
                ny: u32::MAX
            }
        );
    }

    #[test]
    fn from_image_data_rejects_dimensions_below_minimum() {
        let result_1x1 = MtmdBitmap::from_image_data(1, 1, &[0u8; 3]);
        let result_1x2 = MtmdBitmap::from_image_data(1, 2, &[0u8; 6]);
        let result_2x1 = MtmdBitmap::from_image_data(2, 1, &[0u8; 6]);
        let result_0x0 = MtmdBitmap::from_image_data(0, 0, &[]);

        assert_eq!(
            result_1x1.unwrap_err(),
            MtmdBitmapError::ImageDimensionsTooSmall(1, 1)
        );
        assert_eq!(
            result_1x2.unwrap_err(),
            MtmdBitmapError::ImageDimensionsTooSmall(1, 2)
        );
        assert_eq!(
            result_2x1.unwrap_err(),
            MtmdBitmapError::ImageDimensionsTooSmall(2, 1)
        );
        assert_eq!(
            result_0x0.unwrap_err(),
            MtmdBitmapError::ImageDimensionsTooSmall(0, 0)
        );
    }

    #[test]
    fn set_id_changes_id() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        bitmap.set_id("test_image").unwrap();

        assert_eq!(bitmap.id().as_deref(), Some("test_image"));
    }

    #[test]
    fn from_audio_data_creates_valid_bitmap() {
        let audio_samples: Vec<f32> = (0u8..100)
            .map(|index| (f32::from(index) * 0.1).sin())
            .collect();
        let bitmap = MtmdBitmap::from_audio_data(&audio_samples).unwrap();

        assert!(bitmap.is_audio());
    }

    #[test]
    fn data_returns_expected_bytes_for_image() {
        let pixel_data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &pixel_data).unwrap();
        let returned_data = bitmap.data();

        assert_eq!(returned_data, &pixel_data);
    }

    #[test]
    fn id_returns_some_by_default() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();

        assert!(bitmap.id().is_some());
    }

    #[test]
    fn id_returns_custom_value_after_set() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        bitmap.set_id("my_image").unwrap();

        assert_eq!(bitmap.id(), Some("my_image".to_string()));
    }

    #[test]
    fn set_id_with_null_byte_returns_error() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        let result = bitmap.set_id("id\0null");

        assert!(result.is_err());
    }

    #[test]
    fn from_file_success_with_null_bitmap_is_contract_error() {
        let result = unsafe {
            super::from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.unwrap_err(),
            MtmdBitmapError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_mtmd_bitmap_init_from_file",
                detail: "success status contained a null bitmap",
            })
        );
    }

    #[test]
    fn from_file_status_vendored_returned_null_returns_file_unreadable() {
        let result = unsafe {
            super::from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_RETURNED_NULL,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.unwrap_err(),
            MtmdBitmapError::FileUnreadable {
                path: PathBuf::from("/missing/image.png")
            }
        );
    }

    #[test]
    fn from_file_status_error_string_allocation_failed_returns_not_enough_memory() {
        let result = unsafe {
            super::from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(result.unwrap_err(), MtmdBitmapError::NotEnoughMemory);
    }

    #[test]
    fn from_file_status_vendored_threw_cxx_exception_without_a_message_is_a_contract_error() {
        let result = unsafe {
            super::from_file_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.unwrap_err(),
            crate::FfiContractError {
                operation: "llama_rs_mtmd_bitmap_init_from_file",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn from_file_null_context_status_is_a_contract_error() {
        let status = llama_cpp_bindings_sys::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_NULL_CTX_ARG;
        let result = unsafe {
            super::from_file_status_to_result(
                status,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.unwrap_err(),
            MtmdBitmapError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_mtmd_bitmap_init_from_file",
                detail: "was given a null ctx argument",
            })
        );
    }
}
