use std::ffi::CString;
use std::ffi::c_char;
use std::ptr::NonNull;

use crate::model::LlamaModel;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_context_params::MtmdContextParams;
use super::mtmd_encode_error::MtmdEncodeError;
use super::mtmd_init_error::MtmdInitError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;
use super::mtmd_input_text::MtmdInputText;
use super::mtmd_tokenize_error::MtmdTokenizeError;

fn map_tokenize_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdTokenizeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT => {
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_IMAGE_PREPROCESSING_ERROR => {
            Err(MtmdTokenizeError::MediaPreprocessingFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdTokenizeError::UnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdTokenizeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_OUT_OF_MEMORY => {
            Err(MtmdTokenizeError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_string(out_error, "llama_rs_mtmd_tokenize", "reported a thrown C++ exception without an error message") }?;
            Err(MtmdTokenizeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_tokenize",
                detail: "nonzero bitmap count was observed with a null bitmap array",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null output argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG => Err(crate::FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null text argument",
        }
        .into()),
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_mtmd_tokenize",
            code: i64::from(other),
        }
        .into()),
    }
}

fn map_encode_chunk_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk_status,
    vendored_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdEncodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_RETURNED_NONZERO_CODE => {
            Err(MtmdEncodeError::EncodingFailed {
                code: vendored_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdEncodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_OUT_OF_MEMORY => {
            Err(MtmdEncodeError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_encode_chunk",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdEncodeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_mtmd_encode_chunk",
            code: i64::from(other),
        }
        .into()),
    }
}

fn map_init_from_file_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_init_from_file_status,
    out_ctx: *mut llama_cpp_bindings_sys::mtmd_context,
    out_error: *mut c_char,
    mmproj_path: &str,
) -> Result<MtmdContext, MtmdInitError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK => {
            let context = NonNull::new(out_ctx).ok_or_else(|| {
                MtmdInitError::from(crate::FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "success status contained a null multimodal context",
                })
            })?;
            Ok(MtmdContext { context })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_RETURNED_NULL => {
            Err(MtmdInitError::Unloadable {
                path: std::path::PathBuf::from(mmproj_path),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdInitError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_OUT_OF_MEMORY => {
            Err(MtmdInitError::VendoredOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_init_from_file",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdInitError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null mmproj_path argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null text_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null out_ctx argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_mtmd_init_from_file",
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Debug)]
pub struct MtmdContext {
    pub context: NonNull<llama_cpp_bindings_sys::mtmd_context>,
}

unsafe impl Send for MtmdContext {}
unsafe impl Sync for MtmdContext {}

impl MtmdContext {
    /// # Errors
    ///
    /// Returns an [`MtmdInitError`] variant matching the wrapper's status code.
    pub fn init_from_file(
        mmproj_path: &str,
        text_model: &LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = CString::new(mmproj_path)?;
        let ctx_params = llama_cpp_bindings_sys::mtmd_context_params::from(params);

        let mut out_ctx: *mut llama_cpp_bindings_sys::mtmd_context = std::ptr::null_mut();
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                ctx_params,
                &raw mut out_ctx,
                &raw mut out_error,
            )
        };

        map_init_from_file_status(status, out_ctx, out_error, mmproj_path)
    }

    #[must_use]
    pub fn decode_use_non_causal(&self, chunk: &MtmdInputChunk) -> bool {
        unsafe {
            llama_cpp_bindings_sys::mtmd_decode_use_non_causal(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
            )
        }
    }

    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn support_vision(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_support_vision(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn support_audio(&self) -> bool {
        unsafe { llama_cpp_bindings_sys::mtmd_support_audio(self.context.as_ptr()) }
    }

    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let rate =
            unsafe { llama_cpp_bindings_sys::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// # Errors
    ///
    /// Returns an [`MtmdTokenizeError`] variant matching the wrapper's status code.
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new()?;
        let text_cstring = CString::new(text.text)?;
        let input_text = llama_cpp_bindings_sys::mtmd_input_text {
            text: text_cstring.as_ptr(),
            text_len: text_cstring.as_bytes().len(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        let bitmap_ptrs: Vec<*const llama_cpp_bindings_sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|bitmap| bitmap.bitmap.as_ptr().cast_const())
            .collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr().cast_mut(),
                bitmaps.len(),
                &raw mut out_undocumented_return_code,
                &raw mut out_error,
            )
        };

        map_tokenize_status(status, out_undocumented_return_code, out_error)?;
        Ok(chunks)
    }

    /// # Errors
    ///
    /// Returns an [`MtmdEncodeError`] variant matching the wrapper's status code.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk) -> Result<(), MtmdEncodeError> {
        let mut out_vendored_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
                &raw mut out_vendored_return_code,
                &raw mut out_error,
            )
        };

        map_encode_chunk_status(status, out_vendored_return_code, out_error)
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::map_encode_chunk_status;
    use super::map_init_from_file_status;
    use super::map_tokenize_status;
    use crate::mtmd::mtmd_encode_error::MtmdEncodeError;
    use crate::mtmd::mtmd_init_error::MtmdInitError;
    use crate::mtmd::mtmd_tokenize_error::MtmdTokenizeError;

    #[test]
    fn tokenize_status_maps_bitmap_count_mismatch() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        );
    }

    #[test]
    fn tokenize_status_maps_media_preprocessing_failed() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_REPORTED_IMAGE_PREPROCESSING_ERROR,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdTokenizeError::MediaPreprocessingFailed));
    }

    #[test]
    fn tokenize_status_maps_unknown_status_with_value() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_RETURNED_UNDOCUMENTED_NONZERO_CODE,
            42,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdTokenizeError::UnknownStatus { code: 42 }));
    }

    #[test]
    fn tokenize_status_maps_ok_to_unit() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn encode_chunk_status_maps_ok_to_unit() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn encode_chunk_status_maps_encoding_failed_with_code() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_RETURNED_NONZERO_CODE,
            5,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEncodeError::EncodingFailed { code: 5 }));
    }

    #[test]
    fn tokenize_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdTokenizeError::NotEnoughMemory));
    }

    #[test]
    fn tokenize_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_THREW_CXX_EXCEPTION,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_tokenize",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn tokenize_null_bitmaps_status_is_contract_error() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdTokenizeError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_mtmd_tokenize",
                detail: "nonzero bitmap count was observed with a null bitmap array",
            }))
        );
    }

    #[test]
    fn tokenize_unknown_status_is_preserved() {
        let result = map_tokenize_status(255, 0, std::ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdTokenizeError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_mtmd_tokenize",
                code: 255,
            }))
        );
    }

    #[test]
    fn encode_chunk_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEncodeError::NotEnoughMemory));
    }

    #[test]
    fn encode_chunk_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_THREW_CXX_EXCEPTION,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn encode_chunk_unknown_status_is_preserved() {
        let result = map_encode_chunk_status(255, 0, std::ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdEncodeError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_mtmd_encode_chunk",
                code: 255,
            }))
        );
    }

    #[test]
    fn init_from_file_success_with_null_context_is_contract_error() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            MtmdInitError::FfiContract(crate::FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "success status contained a null multimodal context",
            })
        );
    }

    #[test]
    fn init_from_file_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(result.unwrap_err(), MtmdInitError::NotEnoughMemory);
    }

    #[test]
    fn init_from_file_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_THREW_CXX_EXCEPTION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            crate::FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn init_from_file_unknown_status_is_preserved() {
        let result = map_init_from_file_status(
            255,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            MtmdInitError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_mtmd_init_from_file",
                code: 255,
            })
        );
    }
}

#[cfg(test)]
mod ffi_contract_status_tests {
    use super::map_encode_chunk_status;
    use super::map_init_from_file_status;
    use super::map_tokenize_status;
    use crate::mtmd::mtmd_encode_error::MtmdEncodeError;
    use crate::mtmd::mtmd_init_error::MtmdInitError;
    use crate::mtmd::mtmd_tokenize_error::MtmdTokenizeError;
    use std::ptr;

    #[test]
    fn map_tokenize_status_maps_every_contract_status() {
        let outcome_0 = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
        let outcome_1 = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_1.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null output argument",
                }
                .into()
            )
        );
        let outcome_2 = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_2.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null text argument",
                }
                .into()
            )
        );
        let outcome_3 = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_VENDORED_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_3.err(),
            Some(MtmdTokenizeError::VendoredOutOfMemory)
        );
    }

    #[test]
    fn map_encode_chunk_status_maps_every_contract_status() {
        let outcome_0 = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
        let outcome_1 = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_1.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
        let outcome_2 = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_VENDORED_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome_2.err(), Some(MtmdEncodeError::VendoredOutOfMemory));
    }

    #[test]
    fn map_init_from_file_status_maps_every_contract_status() {
        let outcome_0 = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null mmproj_path argument",
                }
                .into()
            )
        );
        let outcome_1 = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome_1.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null text_model argument",
                }
                .into()
            )
        );
        let outcome_2 = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome_2.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null out_ctx argument",
                }
                .into()
            )
        );
        let outcome_3 = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_VENDORED_OUT_OF_MEMORY,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(outcome_3.err(), Some(MtmdInitError::VendoredOutOfMemory));
    }
}
