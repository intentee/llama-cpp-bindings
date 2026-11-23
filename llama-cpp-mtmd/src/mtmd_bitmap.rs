use std::ffi::CString;
use std::ffi::NulError;
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::str::Utf8Error;

use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_audio_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_buf_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_file_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;

use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::optional_mtmd_id::optional_mtmd_id;

/// # Safety
///
/// On `LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK` `out_bitmap` must be the non-null bitmap llama.cpp created, and on
/// `LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message the C++ side
/// allocated; the message is freed here.
unsafe fn from_file_status_to_result(
    status: llama_rs_mtmd_bitmap_init_from_file_status,
    out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap,
    out_error: *mut c_char,
    path: &str,
) -> Result<MtmdBitmap, MtmdError> {
    match status {
        llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_OK => Ok(MtmdBitmap {
            bitmap: unsafe { NonNull::new_unchecked(out_bitmap) },
        }),
        llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            Err(MtmdError::FileUnreadable {
                path: PathBuf::from(path),
            })
        }
        llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::BitmapNotEnoughMemory)
        }
        llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::BitmapLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_bitmap_init_from_file", |message| {
                    MtmdError::BitmapReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_OK` `out_bitmap` must be the non-null bitmap llama.cpp created, and on
/// `LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message the C++ side
/// allocated; the message is freed here.
unsafe fn from_buf_status_to_result(
    status: llama_rs_mtmd_bitmap_init_from_buf_status,
    out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap,
    out_error: *mut c_char,
) -> Result<MtmdBitmap, MtmdError> {
    match status {
        llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_OK => Ok(MtmdBitmap {
            bitmap: unsafe { NonNull::new_unchecked(out_bitmap) },
        }),
        llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_RETURNED_NULL => {
            Err(MtmdError::BitmapDecodeFailed)
        }
        llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::BitmapNotEnoughMemory)
        }
        llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::BitmapLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_bitmap_init_from_buf", |message| {
                    MtmdError::BitmapReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_BITMAP_INIT_OK` `out_bitmap` must be the non-null bitmap llama.cpp created, and on
/// `LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message the C++ side
/// allocated; the message is freed here.
unsafe fn bitmap_init_status_to_result(
    status: llama_rs_mtmd_bitmap_init_status,
    out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap,
    out_error: *mut c_char,
) -> Result<MtmdBitmap, MtmdError> {
    match status {
        llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_OK => Ok(MtmdBitmap {
            bitmap: unsafe { NonNull::new_unchecked(out_bitmap) },
        }),
        llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::BitmapNotEnoughMemory)
        }
        llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::BitmapLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_bitmap_init", |message| {
                    MtmdError::BitmapReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_OK` `out_bitmap` must be the non-null bitmap llama.cpp created, and on
/// `LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null message the C++ side
/// allocated; the message is freed here.
unsafe fn bitmap_init_from_audio_status_to_result(
    status: llama_rs_mtmd_bitmap_init_from_audio_status,
    out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap,
    out_error: *mut c_char,
) -> Result<MtmdBitmap, MtmdError> {
    match status {
        llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_OK => Ok(MtmdBitmap {
            bitmap: unsafe { NonNull::new_unchecked(out_bitmap) },
        }),
        llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::BitmapNotEnoughMemory)
        }
        llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::BitmapLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_bitmap_init_from_audio", |message| {
                    MtmdError::BitmapReported { message }
                }))
        }
    }
}

const RGB_BYTES_PER_PIXEL: u64 = 3;

/// Owns the bitmap and frees it on drop, so it cannot be cloned. Only [`Self::set_id`] writes to
/// it, through `&mut self`, so threads can share it for reading.
#[derive(Debug)]
pub struct MtmdBitmap {
    bitmap: NonNull<llama_cpp_bindings_sys::bindings::mtmd_bitmap>,
}

unsafe impl Send for MtmdBitmap {}
unsafe impl Sync for MtmdBitmap {}

impl MtmdBitmap {
    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to build the audio bitmap.
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdError> {
        let mut out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_audio(
                data.len(),
                data.as_ptr(),
                &raw mut out_bitmap,
                &raw mut out_error,
            )
        };

        unsafe { bitmap_init_from_audio_status_to_result(status, out_bitmap, out_error) }
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to decode the buffer into a bitmap.
    pub fn from_buffer(mtmd_context: &MtmdContext, data: &[u8]) -> Result<Self, MtmdError> {
        let mut out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_buf(
                mtmd_context.as_ptr(),
                data.as_ptr(),
                data.len(),
                &raw mut out_bitmap,
                &raw mut out_error,
            )
        };

        unsafe { from_buf_status_to_result(status, out_bitmap, out_error) }
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when the path cannot be passed to llama.cpp, or when llama.cpp
    /// fails to read the file into a bitmap.
    pub fn from_file(mtmd_context: &MtmdContext, path: &str) -> Result<Self, MtmdError> {
        let path_cstr = CString::new(path).map_err(MtmdError::BitmapPathContainsNul)?;
        let mut out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_file(
                mtmd_context.as_ptr(),
                path_cstr.as_ptr(),
                &raw mut out_bitmap,
                &raw mut out_error,
            )
        };

        unsafe { from_file_status_to_result(status, out_bitmap, out_error, path) }
    }

    /// # Errors
    ///
    /// * `ImageDimensionsTooSmall` - `nx` or `ny` is below the 2x2 minimum
    /// * `ImageDimensionsOverflow` - `nx * ny * 3` does not fit into a buffer length
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * the [`MtmdError`] variant for llama.cpp failing to build the image bitmap
    ///
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdError> {
        if nx < 2 || ny < 2 {
            return Err(MtmdError::ImageDimensionsTooSmall { nx, ny });
        }

        let expected_len = u64::from(nx)
            .checked_mul(u64::from(ny))
            .and_then(|pixels| pixels.checked_mul(RGB_BYTES_PER_PIXEL))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or(MtmdError::ImageDimensionsOverflow { nx, ny })?;

        if data.len() != expected_len {
            return Err(MtmdError::InvalidDataSize);
        }

        let mut out_bitmap: *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init(
                nx,
                ny,
                data.as_ptr(),
                &raw mut out_bitmap,
                &raw mut out_error,
            )
        };

        unsafe { bitmap_init_status_to_result(status, out_bitmap, out_error) }
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *mut llama_cpp_bindings_sys::bindings::mtmd_bitmap {
        self.bitmap.as_ptr()
    }

    /// The bitmap's bytes, or none for a placeholder bitmap, whose data mtmd reports as null.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        let data =
            unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr())
        };

        if data.is_null() {
            return &[];
        }

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// # Errors
    ///
    /// Returns [`Utf8Error`] when the id is not valid UTF-8.
    pub fn id(&self) -> Result<Option<String>, Utf8Error> {
        unsafe {
            optional_mtmd_id(llama_cpp_bindings_sys::bindings::mtmd_bitmap_get_id(
                self.bitmap.as_ptr(),
            ))
        }
    }

    #[must_use]
    pub fn is_audio(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    #[must_use]
    pub fn nx(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    #[must_use]
    pub fn ny(&self) -> u32 {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    pub fn set_id(&mut self, id: &str) -> Result<(), NulError> {
        let id_cstr = CString::new(id)?;
        unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_bitmap_set_id(
                self.bitmap.as_ptr(),
                id_cstr.as_ptr(),
            );
        }

        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::path::PathBuf;
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_audio_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_buf_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_from_file_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_bitmap_init_status;

    use super::bitmap_init_from_audio_status_to_result;
    use super::bitmap_init_status_to_result;
    use super::from_buf_status_to_result;
    use super::from_file_status_to_result;
    use crate::mtmd_bitmap::MtmdBitmap;
    use crate::mtmd_error::MtmdError;

    trait ResolvesOnlyWithoutClone<TMarker> {
        fn resolved() -> bool {
            true
        }
    }

    impl<TValue> ResolvesOnlyWithoutClone<()> for TValue {}

    impl<TValue: Clone> ResolvesOnlyWithoutClone<u8> for TValue {}

    #[test]
    fn a_bitmap_cannot_be_cloned() {
        assert!(<MtmdBitmap as ResolvesOnlyWithoutClone<_>>::resolved());
    }

    #[test]
    fn image_data_makes_an_image_bitmap_of_its_dimensions() {
        let red_pixel: [u8; 3] = [255, 0, 0];
        let image_data: Vec<u8> = red_pixel.repeat(4);
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        assert_eq!(bitmap.nx(), 2);
        assert_eq!(bitmap.ny(), 2);
        assert!(!bitmap.is_audio());
    }

    #[test]
    fn image_data_shorter_than_its_dimensions_is_an_invalid_data_size() {
        let too_short = vec![0u8; 5];

        assert_eq!(
            MtmdBitmap::from_image_data(2, 2, &too_short).unwrap_err(),
            MtmdError::InvalidDataSize
        );
    }

    #[test]
    fn dimensions_whose_rgb_buffer_size_overflows_are_rejected() {
        assert_eq!(
            MtmdBitmap::from_image_data(u32::MAX, u32::MAX, &[0u8; 12]).unwrap_err(),
            MtmdError::ImageDimensionsOverflow {
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
            MtmdError::ImageDimensionsTooSmall { nx: 1, ny: 1 }
        );
        assert_eq!(
            result_1x2.unwrap_err(),
            MtmdError::ImageDimensionsTooSmall { nx: 1, ny: 2 }
        );
        assert_eq!(
            result_2x1.unwrap_err(),
            MtmdError::ImageDimensionsTooSmall { nx: 2, ny: 1 }
        );
        assert_eq!(
            result_0x0.unwrap_err(),
            MtmdError::ImageDimensionsTooSmall { nx: 0, ny: 0 }
        );
    }

    #[test]
    fn set_id_replaces_the_bitmap_id() {
        let image_data = vec![0u8; 12];
        let mut bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        bitmap.set_id("test_image").unwrap();

        assert_eq!(bitmap.id(), Ok(Some("test_image".to_owned())));
    }

    #[test]
    fn audio_samples_make_an_audio_bitmap() {
        let audio_samples: Vec<f32> = (0u8..100)
            .map(|index| (f32::from(index) * 0.1).sin())
            .collect();
        let bitmap = MtmdBitmap::from_audio_data(&audio_samples).unwrap();

        assert!(bitmap.is_audio());
    }

    #[test]
    fn a_placeholder_bitmap_without_samples_holds_no_data() {
        let bitmap = MtmdBitmap::from_audio_data(&[]).unwrap();

        assert_eq!(bitmap.data(), &[] as &[u8]);
    }

    #[test]
    fn a_bitmap_hands_back_the_pixel_data_it_was_built_from() {
        let pixel_data: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &pixel_data).unwrap();
        let returned_data = bitmap.data();

        assert_eq!(returned_data, &pixel_data);
    }

    #[test]
    fn a_new_bitmap_has_an_empty_id() {
        let image_data = vec![0u8; 12];
        let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();

        assert_eq!(bitmap.id(), Ok(Some(String::new())));
    }

    #[test]
    fn set_id_reports_a_nul_byte_in_the_id() {
        let image_data = vec![0u8; 12];
        let mut bitmap = MtmdBitmap::from_image_data(2, 2, &image_data).unwrap();
        let result = bitmap.set_id("id\0null");

        assert_eq!(result, Err(CString::new("id\0null").unwrap_err()));
    }

    #[test]
    fn from_file_status_llama_cpp_returned_null_returns_file_unreadable() {
        let result = unsafe {
            from_file_status_to_result(
                llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.unwrap_err(),
            MtmdError::FileUnreadable {
                path: PathBuf::from("/missing/image.png")
            }
        );
    }

    #[test]
    fn from_file_status_error_string_allocation_failed_returns_not_enough_memory() {
        let result = unsafe {
            from_file_status_to_result(
                llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
                "/missing/image.png",
            )
        };

        assert_eq!(result.unwrap_err(), MtmdError::BitmapNotEnoughMemory);
    }

    #[test]
    fn from_file_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            from_file_status_to_result(
                llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
                "",
            )
        };
        assert_eq!(outcome.err(), Some(MtmdError::BitmapLlamaCppOutOfMemory));
    }

    #[test]
    fn from_file_status_llama_cpp_threw_cxx_exception_carries_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"image loader threw".as_ptr())
        };

        let result = unsafe {
            from_file_status_to_result(
                llama_rs_mtmd_bitmap_init_from_file_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
                "/missing/image.png",
            )
        };

        assert_eq!(
            result.err(),
            Some(MtmdError::BitmapReported {
                message: "image loader threw".to_owned(),
            })
        );
    }

    #[test]
    fn success_status_adopts_the_bitmap_llama_cpp_produced() {
        let pixels = [0u8; 12];
        let produced =
            unsafe { llama_cpp_bindings_sys::bindings::mtmd_bitmap_init(2, 2, pixels.as_ptr()) };

        let result = unsafe {
            from_buf_status_to_result(
                llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_OK,
                produced,
                ptr::null_mut(),
            )
        };

        assert_eq!(result.map(|adopted| adopted.nx()), Ok(2));
    }

    #[test]
    fn returned_null_status_reports_a_decode_failure() {
        let result = unsafe {
            from_buf_status_to_result(
                llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_RETURNED_NULL,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.err(), Some(MtmdError::BitmapDecodeFailed));
    }

    #[test]
    fn error_string_allocation_failure_reports_not_enough_memory() {
        let result = unsafe {
            from_buf_status_to_result(
                llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_ERROR_STRING_ALLOCATION_FAILED,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.err(), Some(MtmdError::BitmapNotEnoughMemory));
    }

    #[test]
    fn out_of_memory_status_reports_that_llama_cpp_ran_out_of_memory() {
        let result = unsafe {
            from_buf_status_to_result(
                llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_OUT_OF_MEMORY,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };

        assert_eq!(result.err(), Some(MtmdError::BitmapLlamaCppOutOfMemory));
    }

    #[test]
    fn exception_status_surfaces_the_reported_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"buffer decoder threw".as_ptr())
        };
        let result = unsafe {
            from_buf_status_to_result(
                llama_rs_mtmd_bitmap_init_from_buf_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_BUF_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
        };

        assert_eq!(
            result.err(),
            Some(MtmdError::BitmapReported {
                message: "buffer decoder threw".to_owned(),
            })
        );
    }

    #[test]
    fn image_initialization_memory_failures_map_to_their_errors() {
        let statuses = [
            llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| {
                unsafe { bitmap_init_status_to_result(status, ptr::null_mut(), ptr::null_mut()) }
                    .err()
            }),
            [
                Some(MtmdError::BitmapNotEnoughMemory),
                Some(MtmdError::BitmapLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn audio_initialization_memory_failures_map_to_their_errors() {
        let statuses = [
            llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| {
                unsafe {
                    bitmap_init_from_audio_status_to_result(
                        status,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                }
                .err()
            }),
            [
                Some(MtmdError::BitmapNotEnoughMemory),
                Some(MtmdError::BitmapLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn an_exception_thrown_while_initializing_an_image_bitmap_reports_its_message() {
        let image_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"image threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                bitmap_init_status_to_result(
                    llama_rs_mtmd_bitmap_init_status::LLAMA_RS_MTMD_BITMAP_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    image_error,
                )
            }
            .err(),
            Some(MtmdError::BitmapReported {
                message: "image threw".to_owned(),
            })
        );
    }

    #[test]
    fn an_exception_thrown_while_initializing_an_audio_bitmap_reports_its_message() {
        let audio_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"audio threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                bitmap_init_from_audio_status_to_result(
                    llama_rs_mtmd_bitmap_init_from_audio_status::LLAMA_RS_MTMD_BITMAP_INIT_FROM_AUDIO_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    audio_error,
                )
            }
            .err(),
            Some(MtmdError::BitmapReported {
                message: "audio threw".to_owned(),
            })
        );
    }
}
