use std::ffi::CString;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunks_init_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize_from_parts_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize_status;
use llama_cpp_bindings_sys::cpp_allocated_string::CppAllocatedString;
use llama_cpp_context::llama_context::LlamaContext;

use crate::evaluated_mtmd_input_chunk::EvaluatedMtmdInputChunk;
use crate::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use crate::mtmd_bitmap::MtmdBitmap;
use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_ffi_input_part::MtmdFfiInputPart;
use crate::mtmd_input_chunk_view::MtmdInputChunkView;
use crate::mtmd_input_part::MtmdInputPart;
use crate::mtmd_input_text::MtmdInputText;

/// # Safety
///
/// On `LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK` `out_chunks` must be the non-null chunk list llama.cpp
/// created, and on `LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must
/// be the non-null message the C++ side allocated; the message is freed here.
unsafe fn input_chunks_init_status_to_result(
    status: llama_rs_mtmd_input_chunks_init_status,
    out_chunks: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunks,
    out_error: *mut c_char,
) -> Result<NonNull<llama_cpp_bindings_sys::bindings::mtmd_input_chunks>, MtmdError> {
    match status {
        llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK => {
            Ok(unsafe { NonNull::new_unchecked(out_chunks) })
        }
        llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::InputChunksNotEnoughMemory)
        }
        llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::InputChunksLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_input_chunks_init", |message| {
                    MtmdError::InputChunksReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null
/// message the C++ side allocated; it is freed here.
unsafe fn map_tokenize_status(
    status: llama_rs_mtmd_tokenize_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    match status {
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_OK => Ok(()),
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT => {
            Err(MtmdError::BitmapCountDoesNotMatchMarkerCount)
        }
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdError::TokenizeUnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::TokenizeNotEnoughMemory)
        }
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_tokenize", |message| {
                    MtmdError::TokenizeReported { message }
                }))
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; it is freed here.
unsafe fn map_tokenize_from_parts_status(
    status: llama_rs_mtmd_tokenize_from_parts_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    match status {
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK => Ok(()),
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART => {
            Err(MtmdError::LlamaCppReportedMalformedPartList {
                operation: "llama_rs_mtmd_tokenize_from_parts",
            })
        }
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdError::TokenizeFromPartsUnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::TokenizeNotEnoughMemory)
        }
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            Err(unsafe { CppAllocatedString::from_raw(out_error) }
                .into_reported_error("llama_rs_mtmd_tokenize_from_parts", |message| {
                    MtmdError::TokenizeReported { message }
                }))
        }
    }
}

/// Chunks tokenized by one projector, which they borrow so that they are only ever fitted,
/// encoded and evaluated with it.
#[derive(Debug)]
pub struct MtmdInputChunks<'context> {
    chunks: NonNull<llama_cpp_bindings_sys::bindings::mtmd_input_chunks>,
    mtmd_context: &'context MtmdContext,
}

impl<'context> MtmdInputChunks<'context> {
    /// An empty chunk list for the chunks `mtmd_context` tokenizes.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to create the chunk list.
    pub fn new(mtmd_context: &'context MtmdContext) -> Result<Self, MtmdError> {
        let mut out_chunks: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunks =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunks_init(
                &raw mut out_chunks,
                &raw mut out_error,
            )
        };

        Ok(Self {
            chunks: unsafe { input_chunks_init_status_to_result(status, out_chunks, out_error) }?,
            mtmd_context,
        })
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when the chunk list cannot be created, the text holds a nul byte,
    /// or llama.cpp fails to tokenize the text and bitmaps.
    pub fn tokenize(
        mtmd_context: &'context MtmdContext,
        MtmdInputText {
            text,
            add_special,
            parse_special,
        }: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<Self, MtmdError> {
        let chunks = Self::new(mtmd_context)
            .map_err(Box::new)
            .map_err(MtmdError::TokenizeInputChunks)?;
        let text_cstring = CString::new(text).map_err(MtmdError::InputTextContainsNul)?;
        let input_text = llama_cpp_bindings_sys::bindings::mtmd_input_text {
            text: text_cstring.as_ptr(),
            text_len: text_cstring.as_bytes().len(),
            add_special,
            parse_special,
        };

        let bitmap_ptrs: Vec<*const llama_cpp_bindings_sys::bindings::mtmd_bitmap> = bitmaps
            .iter()
            .map(|bitmap| bitmap.as_ptr().cast_const())
            .collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize(
                mtmd_context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_ptrs.as_ptr(),
                bitmaps.len(),
                &raw mut out_undocumented_return_code,
                &raw mut out_error,
            )
        };

        unsafe { map_tokenize_status(status, out_undocumented_return_code, out_error) }?;
        Ok(chunks)
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when the chunk list cannot be created, a text part holds a nul
    /// byte, or llama.cpp fails to tokenize the parts.
    pub fn tokenize_from_parts(
        mtmd_context: &'context MtmdContext,
        parts: &[MtmdInputPart<'_>],
        add_special: bool,
    ) -> Result<Self, MtmdError> {
        let chunks = Self::new(mtmd_context)
            .map_err(Box::new)
            .map_err(MtmdError::TokenizeInputChunks)?;
        let ffi_input_parts = parts
            .iter()
            .map(MtmdFfiInputPart::try_from)
            .collect::<Result<Vec<MtmdFfiInputPart<'_>>, _>>()
            .map_err(MtmdError::InputTextContainsNul)?;
        let ffi_parts: Vec<llama_cpp_bindings_sys::bindings::mtmd_input_part> = ffi_input_parts
            .iter()
            .map(MtmdFfiInputPart::ffi_part)
            .collect();
        let ffi_part_pointers: Vec<*const llama_cpp_bindings_sys::bindings::mtmd_input_part> =
            ffi_parts.iter().map(ptr::from_ref).collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize_from_parts(
                mtmd_context.as_ptr(),
                chunks.chunks.as_ptr(),
                ffi_part_pointers.as_ptr(),
                ffi_part_pointers.len(),
                add_special,
                &raw mut out_undocumented_return_code,
                &raw mut out_error,
            )
        };

        unsafe { map_tokenize_from_parts_status(status, out_undocumented_return_code, out_error) }?;
        Ok(chunks)
    }

    #[must_use]
    pub fn chunks(&self) -> Vec<MtmdInputChunkView<'_, 'context>> {
        (0..self.len())
            .map(|index| unsafe { self.chunk_within_count(index) })
            .collect()
    }

    /// Checks every chunk with [`Self::fit_in`] and its positions with
    /// [`FittedMtmdInputChunks::eval_each`] before evaluating any of them, so a chunk that cannot
    /// be decoded leaves the KV cache untouched.
    ///
    /// # Errors
    ///
    /// Returns any error of [`Self::fit_in`] or [`FittedMtmdInputChunks::eval_each`], or
    /// [`MtmdError::EvalFailed`] if any encoding or decoding operation fails.
    pub fn eval_chunks(
        &self,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
        let fitted_chunks = self.fit_in(llama_ctx, n_batch, seq_id)?;

        fitted_chunks
            .eval_each(llama_ctx, start_position, seq_id, n_batch, logits_last)?
            .try_fold(start_position, |_position_before_chunk, evaluated_chunk| {
                evaluated_chunk.map(|EvaluatedMtmdInputChunk { next_position, .. }| next_position)
            })
    }

    /// Checks `llama_ctx`, `n_batch` and `seq_id` with [`MtmdContext::batch_tokens_in`], then
    /// every chunk with [`FittedMtmdInputChunks::fit`] for decodes of `n_batch` in `llama_ctx`.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdContext::batch_tokens_in`], then any error of
    /// [`FittedMtmdInputChunks::fit`].
    pub fn fit_in(
        &self,
        llama_ctx: &LlamaContext,
        n_batch: i32,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
    ) -> Result<FittedMtmdInputChunks<'_, 'context>, MtmdError> {
        let batch_tokens = self
            .mtmd_context
            .batch_tokens_in(llama_ctx, n_batch, seq_id)?;

        FittedMtmdInputChunks::fit(self.chunks(), llama_ctx, batch_tokens)
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunkView<'_, 'context>> {
        (index < self.len()).then(|| unsafe { self.chunk_within_count(index) })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// The projector that tokenized these chunks.
    #[must_use]
    pub const fn mtmd_context(&self) -> &'context MtmdContext {
        self.mtmd_context
    }

    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    /// # Safety
    ///
    /// `index` must be below [`Self::len`], where llama.cpp always holds a chunk.
    unsafe fn chunk_within_count(&self, index: usize) -> MtmdInputChunkView<'_, 'context> {
        let chunk = unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_input_chunks_get(self.chunks.as_ptr(), index)
        };

        unsafe {
            MtmdInputChunkView::from_raw(
                NonNull::new_unchecked(chunk.cast_mut()),
                self.mtmd_context,
            )
        }
    }
}

impl Drop for MtmdInputChunks<'_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunks_init_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize_from_parts_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_tokenize_status;

    use super::input_chunks_init_status_to_result;
    use super::map_tokenize_from_parts_status;
    use super::map_tokenize_status;
    use crate::mtmd_error::MtmdError;

    #[test]
    fn chunk_list_initialization_memory_failures_map_to_their_errors() {
        let statuses = [
            llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| {
                unsafe {
                    input_chunks_init_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                }
                .err()
            }),
            [
                Some(MtmdError::InputChunksNotEnoughMemory),
                Some(MtmdError::InputChunksLlamaCppOutOfMemory),
            ]
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"allocation threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                input_chunks_init_status_to_result(
                    llama_rs_mtmd_input_chunks_init_status::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            }
            .err(),
            Some(MtmdError::InputChunksReported {
                message: "allocation threw".to_owned(),
            })
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_ok_to_unit() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK,
                    0,
                    ptr::null_mut(),
                )
            },
            Ok(())
        );
    }

    #[test]
    fn tokenize_from_parts_status_reports_a_media_preprocessing_or_tokenization_failure() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
                    0,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_an_undocumented_code_with_its_value() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
                    42,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::TokenizeFromPartsUnknownStatus { code: 42 })
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_string_allocation_failed_to_not_enough_memory() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED,
                    0,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::TokenizeNotEnoughMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_reports_llama_cpp_running_out_of_memory() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY,
                    0,
                    ptr::null_mut(),
                )
            },
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_a_thrown_exception_with_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"parts tokenizer threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                )
            }
            .err(),
            Some(MtmdError::TokenizeReported {
                message: "parts tokenizer threw".to_owned(),
            })
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_malformed_part_list() {
        assert_eq!(
            unsafe {
                map_tokenize_from_parts_status(
                    llama_rs_mtmd_tokenize_from_parts_status::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART,
                    0,
                    ptr::null_mut(),
                )
            }
            .err(),
            Some(
                MtmdError::LlamaCppReportedMalformedPartList {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                }
            )
        );
    }

    #[test]
    fn tokenize_status_reports_a_bitmap_count_that_does_not_match_the_marker_count() {
        let result = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::BitmapCountDoesNotMatchMarkerCount));
    }

    #[test]
    fn tokenize_status_reports_a_media_preprocessing_or_tokenization_failure() {
        let result = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(
            result,
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        );
    }

    #[test]
    fn tokenize_status_maps_an_undocumented_code_with_its_value() {
        let result = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
                42,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::TokenizeUnknownStatus { code: 42 }));
    }

    #[test]
    fn tokenize_status_maps_ok_to_unit() {
        let result = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_OK,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn tokenize_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::TokenizeNotEnoughMemory));
    }

    #[test]
    fn map_tokenize_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            map_tokenize_status(
                llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(MtmdError::TokenizeLlamaCppOutOfMemory));
    }

    #[test]
    fn map_tokenize_status_maps_a_thrown_exception_with_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"tokenizer threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                map_tokenize_status(
                    llama_rs_mtmd_tokenize_status::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                )
            },
            Err(MtmdError::TokenizeReported {
                message: "tokenizer threw".to_owned(),
            })
        );
    }
}
