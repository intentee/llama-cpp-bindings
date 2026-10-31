use std::ffi::CString;
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;
use std::ptr::NonNull;

use enumflags2::BitFlags;

use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;

use super::mtmd_bitmap::MtmdBitmap;
use super::mtmd_capability::MtmdCapability;
use super::mtmd_context_params::MtmdContextParams;
use super::mtmd_encode_error::MtmdEncodeError;
use super::mtmd_ffi_input_part::MtmdFfiInputPart;
use super::mtmd_init_error::MtmdInitError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;
use super::mtmd_input_part::MtmdInputPart;
use super::mtmd_input_text::MtmdInputText;
use super::mtmd_tokenize_error::MtmdTokenizeError;
use crate::model::LlamaModel;

fn capability_flags(
    llama_cpp_bindings_sys::mtmd_helper_cap {
        chat,
        vision,
        audio,
        video,
        audio_stream,
        text_gen,
        audio_gen,
        audio_gen_spk_ref,
    }: llama_cpp_bindings_sys::mtmd_helper_cap,
) -> BitFlags<MtmdCapability> {
    [
        (chat, MtmdCapability::Chat),
        (vision, MtmdCapability::Vision),
        (audio, MtmdCapability::Audio),
        (video, MtmdCapability::Video),
        (audio_stream, MtmdCapability::AudioStream),
        (text_gen, MtmdCapability::TextGeneration),
        (audio_gen, MtmdCapability::AudioGeneration),
        (
            audio_gen_spk_ref,
            MtmdCapability::AudioGenerationSpeakerReference,
        ),
    ]
    .into_iter()
    .filter(|(is_capable, _capability)| *is_capable)
    .map(|(_is_capable, capability)| capability)
    .collect()
}

fn map_tokenize_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdTokenizeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT => {
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdTokenizeError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdTokenizeError::UnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdTokenizeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdTokenizeError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { read_and_free_cpp_string(out_error, "llama_rs_mtmd_tokenize", "reported a thrown C++ exception without an error message") }?;
            Err(MtmdTokenizeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize",
                detail: "nonzero bitmap count was observed with a null bitmap array",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG => Err(FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG => Err(FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null output argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG => Err(FfiContractError {
            operation: "llama_rs_mtmd_tokenize",
            detail: "was given a null text argument",
        }
        .into()),
        other => Err(FfiStatusError {
            operation: "llama_rs_mtmd_tokenize",
            code: i64::from(other),
        }
        .into()),
    }
}

fn map_tokenize_from_parts_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_from_parts_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdTokenizeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                detail: "llama.cpp reported a malformed part list: a part holding both or neither of text and bitmap, a text part with a null text pointer, or bitmaps that do not line up with their parts",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdTokenizeError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdTokenizeError::UnknownTokenizeFromPartsStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdTokenizeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdTokenizeError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_tokenize_from_parts",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdTokenizeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_CTX_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_OUTPUT_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                detail: "was given a null output argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PARTS_ARG_WHEN_NUM_PARTS_NONZERO => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                detail: "nonzero part count was observed with a null part array",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PART_ELEMENT => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                detail: "was given a null element in the parts argument",
            }
            .into())
        }
        other => Err(FfiStatusError {
            operation: "llama_rs_mtmd_tokenize_from_parts",
            code: i64::from(other),
        }
        .into()),
    }
}

fn map_encode_chunk_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk_status,
    llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdEncodeError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            Err(MtmdEncodeError::EncodingFailed {
                code: llama_cpp_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdEncodeError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdEncodeError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_encode_chunk",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdEncodeError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG => Err(FfiContractError {
            operation: "llama_rs_mtmd_encode_chunk",
            detail: "was given a null ctx argument",
        }
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        other => Err(FfiStatusError {
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
                MtmdInitError::from(FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "success status contained a null multimodal context",
                })
            })?;
            Ok(MtmdContext { context })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_RETURNED_NULL => {
            Err(MtmdInitError::Unloadable {
                path: PathBuf::from(mmproj_path),
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdInitError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdInitError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
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
            Err(FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null mmproj_path argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null text_model argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG => {
            Err(FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "was given a null out_ctx argument",
            }
            .into())
        }
        other => Err(FfiStatusError {
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
        let ctx_params = llama_cpp_bindings_sys::mtmd_context_params::try_from(params)?;

        let mut out_ctx: *mut llama_cpp_bindings_sys::mtmd_context = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();

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
    pub fn capabilities(&self, model: &LlamaModel) -> BitFlags<MtmdCapability> {
        capability_flags(unsafe {
            llama_cpp_bindings_sys::mtmd_helper_get_capabilities(
                model.model.as_ptr(),
                self.context.as_ptr(),
            )
        })
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
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr(),
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
    /// Returns an [`MtmdTokenizeError`] variant matching the wrapper's status code.
    pub fn tokenize_from_parts(
        &self,
        parts: &[MtmdInputPart<'_>],
        add_special: bool,
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new()?;
        let ffi_input_parts = parts
            .iter()
            .map(MtmdFfiInputPart::try_from)
            .collect::<Result<Vec<MtmdFfiInputPart<'_>>, _>>()?;
        let ffi_parts: Vec<llama_cpp_bindings_sys::mtmd_input_part> = ffi_input_parts
            .iter()
            .map(MtmdFfiInputPart::ffi_part)
            .collect();
        let ffi_part_pointers: Vec<*const llama_cpp_bindings_sys::mtmd_input_part> =
            ffi_parts.iter().map(ptr::from_ref).collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_from_parts(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                ffi_part_pointers.as_ptr(),
                ffi_part_pointers.len(),
                add_special,
                &raw mut out_undocumented_return_code,
                &raw mut out_error,
            )
        };

        map_tokenize_from_parts_status(status, out_undocumented_return_code, out_error)?;
        Ok(chunks)
    }

    /// # Errors
    ///
    /// Returns an [`MtmdEncodeError`] variant matching the wrapper's status code.
    pub fn encode_chunk(&self, chunk: &MtmdInputChunk) -> Result<(), MtmdEncodeError> {
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        map_encode_chunk_status(status, out_llama_cpp_return_code, out_error)
    }
}

impl Drop for MtmdContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod map_tokenize_from_parts_status_tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;
    use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;
    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::map_tokenize_from_parts_status;
    use crate::mtmd::mtmd_tokenize_error::MtmdTokenizeError;

    #[test]
    fn tokenize_from_parts_status_maps_ok_to_unit() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK,
                0,
                ptr::null_mut(),
            ),
            Ok(())
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_preprocessing_or_tokenization_failure() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
                0,
                ptr::null_mut(),
            ),
            Err(MtmdTokenizeError::MediaPreprocessingOrTokenizationFailed)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_an_undocumented_code_with_its_value() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
                42,
                ptr::null_mut(),
            ),
            Err(MtmdTokenizeError::UnknownTokenizeFromPartsStatus { code: 42 })
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_string_allocation_failed_to_not_enough_memory() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            ),
            Err(MtmdTokenizeError::NotEnoughMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_llama_cpp_out_of_memory() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            ),
            Err(MtmdTokenizeError::LlamaCppOutOfMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_a_thrown_exception_with_its_message() {
        let expected_message = unsafe {
            read_and_free_cpp_string(
                wrapper_allocated_error_message_pointer(),
                "llama_rs_mtmd_tokenize_from_parts",
                "reported a thrown C++ exception without an error message",
            )
        };

        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                wrapper_allocated_error_message_pointer(),
            )
            .err(),
            expected_message
                .ok()
                .map(|message| MtmdTokenizeError::Reported { message })
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_llama_cpp_reported_malformed_part() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "llama.cpp reported a malformed part list: a part holding both or neither of text and bitmap, a text part with a null text pointer, or bitmaps that do not line up with their parts",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_null_part_element() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PART_ELEMENT,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null element in the parts argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_llama_cpp_threw_cxx_exception_without_a_message() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "reported a thrown C++ exception without an error message",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_null_ctx_arg() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_CTX_ARG,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_null_output_arg() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_OUTPUT_ARG,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null output argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_maps_null_parts_arg_when_num_parts_nonzero() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PARTS_ARG_WHEN_NUM_PARTS_NONZERO,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "nonzero part count was observed with a null part array",
                }
                .into()
            )
        );
        assert_eq!(
            map_tokenize_from_parts_status(255, 0, ptr::null_mut()).err(),
            Some(MtmdTokenizeError::FfiStatus(FfiStatusError {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                code: 255,
            }))
        );
    }
}

#[cfg(test)]
mod map_tokenize_status_tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::map_tokenize_status;
    use crate::mtmd::mtmd_tokenize_error::MtmdTokenizeError;

    #[test]
    fn tokenize_status_maps_bitmap_count_mismatch() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdTokenizeError::BitmapCountDoesNotMatchMarkerCount)
        );
    }

    #[test]
    fn tokenize_status_maps_preprocessing_or_tokenization_failure() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdTokenizeError::MediaPreprocessingOrTokenizationFailed)
        );
    }

    #[test]
    fn tokenize_status_maps_unknown_status_with_value() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
            42,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdTokenizeError::UnknownStatus { code: 42 }));
    }

    #[test]
    fn tokenize_status_maps_ok_to_unit() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn tokenize_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdTokenizeError::NotEnoughMemory));
    }

    #[test]
    fn tokenize_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiContractError {
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
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdTokenizeError::FfiContract(FfiContractError {
                operation: "llama_rs_mtmd_tokenize",
                detail: "nonzero bitmap count was observed with a null bitmap array",
            }))
        );
    }

    #[test]
    fn tokenize_unknown_status_is_preserved() {
        let result = map_tokenize_status(255, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdTokenizeError::FfiStatus(FfiStatusError {
                operation: "llama_rs_mtmd_tokenize",
                code: 255,
            }))
        );
    }

    #[test]
    fn map_tokenize_status_maps_null_ctx_arg() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_maps_null_output_arg() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null output argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_maps_null_text_arg() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null text argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_maps_llama_cpp_out_of_memory() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdTokenizeError::LlamaCppOutOfMemory));
    }

    #[test]
    fn map_tokenize_status_maps_a_thrown_exception_with_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"tokenizer threw".as_ptr()) };

        assert_eq!(
            map_tokenize_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            ),
            Err(MtmdTokenizeError::Reported {
                message: "tokenizer threw".to_owned(),
            })
        );
    }
}

#[cfg(test)]
mod map_encode_chunk_status_tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::map_encode_chunk_status;
    use crate::mtmd::mtmd_encode_error::MtmdEncodeError;

    #[test]
    fn encode_chunk_status_maps_ok_to_unit() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn encode_chunk_status_maps_encoding_failed_with_code() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_RETURNED_NONZERO_CODE,
            5,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEncodeError::EncodingFailed { code: 5 }));
    }

    #[test]
    fn encode_chunk_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEncodeError::NotEnoughMemory));
    }

    #[test]
    fn encode_chunk_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiContractError {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn encode_chunk_unknown_status_is_preserved() {
        let result = map_encode_chunk_status(255, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdEncodeError::FfiStatus(FfiStatusError {
                operation: "llama_rs_mtmd_encode_chunk",
                code: 255,
            }))
        );
    }

    #[test]
    fn map_encode_chunk_status_maps_null_ctx_arg() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_encode_chunk_status_maps_null_chunk_arg() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_encode_chunk_status_maps_llama_cpp_out_of_memory() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdEncodeError::LlamaCppOutOfMemory));
    }

    #[test]
    fn map_encode_chunk_status_maps_a_thrown_exception_with_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"encoder threw".as_ptr()) };

        assert_eq!(
            map_encode_chunk_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                out_error,
            ),
            Err(MtmdEncodeError::Reported {
                message: "encoder threw".to_owned(),
            })
        );
    }
}

#[cfg(test)]
mod map_init_from_file_status_tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_contract_error::FfiContractError;
    use llama_cpp_ffi_status::ffi_status_error::FfiStatusError;

    use super::map_init_from_file_status;
    use crate::mtmd::mtmd_init_error::MtmdInitError;

    #[test]
    fn init_from_file_success_with_null_context_is_contract_error() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_OK,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            MtmdInitError::FfiContract(FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "success status contained a null multimodal context",
            })
        );
    }

    #[test]
    fn init_from_file_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_ERROR_STRING_ALLOCATION_FAILED,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(result.unwrap_err(), MtmdInitError::NotEnoughMemory);
    }

    #[test]
    fn init_from_file_status_maps_cxx_exception_to_without_a_message_is_a_contract_error() {
        let result = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            ptr::null_mut(),
            ptr::null_mut(),
            "mmproj.gguf",
        );

        assert_eq!(
            result.unwrap_err(),
            FfiContractError {
                operation: "llama_rs_mtmd_init_from_file",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into()
        );
    }

    #[test]
    fn init_from_file_unknown_status_is_preserved() {
        let result =
            map_init_from_file_status(255, ptr::null_mut(), ptr::null_mut(), "mmproj.gguf");

        assert_eq!(
            result.unwrap_err(),
            MtmdInitError::FfiStatus(FfiStatusError {
                operation: "llama_rs_mtmd_init_from_file",
                code: 255,
            })
        );
    }

    #[test]
    fn map_init_from_file_status_maps_null_mmproj_path_arg() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_MMPROJ_PATH_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null mmproj_path argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_init_from_file_status_maps_null_text_model_arg() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_TEXT_MODEL_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null text_model argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_init_from_file_status_maps_null_out_ctx_arg() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_NULL_OUT_CTX_ARG,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiContractError {
                    operation: "llama_rs_mtmd_init_from_file",
                    detail: "was given a null out_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_init_from_file_status_maps_llama_cpp_out_of_memory() {
        let outcome = map_init_from_file_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_OUT_OF_MEMORY,
            ptr::null_mut(),
            ptr::null_mut(),
            "",
        );
        assert_eq!(outcome.err(), Some(MtmdInitError::LlamaCppOutOfMemory));
    }

    #[test]
    fn map_init_from_file_status_maps_a_thrown_exception_with_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"projector threw".as_ptr()) };

        assert_eq!(
            map_init_from_file_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_INIT_FROM_FILE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
                "mmproj.gguf",
            )
            .err(),
            Some(MtmdInitError::Reported {
                message: "projector threw".to_owned(),
            })
        );
    }
}

#[cfg(test)]
mod capability_flags_tests {
    use enumflags2::BitFlags;

    use super::capability_flags;
    use crate::mtmd::mtmd_capability::MtmdCapability;

    const NO_CAPABILITIES: llama_cpp_bindings_sys::mtmd_helper_cap =
        llama_cpp_bindings_sys::mtmd_helper_cap {
            chat: false,
            vision: false,
            audio: false,
            video: false,
            audio_stream: false,
            text_gen: false,
            audio_gen: false,
            audio_gen_spk_ref: false,
        };

    #[test]
    fn a_model_without_capabilities_has_no_flags() {
        assert_eq!(capability_flags(NO_CAPABILITIES), BitFlags::empty());
    }

    #[test]
    fn every_capability_of_llama_cpp_maps_to_its_own_flag() {
        let all_capabilities = llama_cpp_bindings_sys::mtmd_helper_cap {
            chat: true,
            vision: true,
            audio: true,
            video: true,
            audio_stream: true,
            text_gen: true,
            audio_gen: true,
            audio_gen_spk_ref: true,
        };

        assert_eq!(
            capability_flags(all_capabilities),
            BitFlags::<MtmdCapability>::all()
        );
    }

    #[test]
    fn a_text_to_speech_model_generates_audio_but_cannot_chat() {
        let text_to_speech = llama_cpp_bindings_sys::mtmd_helper_cap {
            audio_gen: true,
            audio_gen_spk_ref: true,
            ..NO_CAPABILITIES
        };

        assert_eq!(
            capability_flags(text_to_speech),
            MtmdCapability::AudioGeneration | MtmdCapability::AudioGenerationSpeakerReference
        );
    }
}
