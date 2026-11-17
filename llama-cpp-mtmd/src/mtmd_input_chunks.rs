use std::ffi::CString;
use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;

use crate::mtmd_bitmap::MtmdBitmap;
use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_eval_limits::MtmdEvalLimits;
use crate::mtmd_ffi_input_part::MtmdFfiInputPart;
use crate::mtmd_input_chunk::MtmdInputChunk;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::mtmd_input_part::MtmdInputPart;
use crate::mtmd_input_text::MtmdInputText;

fn input_chunks_init_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_input_chunks_init_status,
    out_chunks: *mut llama_cpp_bindings_sys::mtmd_input_chunks,
    out_error: *mut c_char,
) -> Result<MtmdInputChunks, MtmdError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_mtmd_input_chunks_init");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK => NonNull::new(out_chunks)
            .map(|chunks| MtmdInputChunks { chunks })
            .ok_or_else(|| {
                OPERATION
                    .contract_violated("success status contained null chunks")
                    .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_NULL_OUT_CHUNKS_ARG => {
            Err(OPERATION
                .contract_violated("was given a null out_chunks argument")
                .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::InputChunksNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::InputChunksLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(out_error) }?;

            Err(MtmdError::InputChunksReported { message })
        }
        other => Err(OPERATION.unknown_status(i64::from(other)).into()),
    }
}

fn map_tokenize_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_mtmd_tokenize");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT => {
            Err(MtmdError::BitmapCountDoesNotMatchMarkerCount)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdError::TokenizeUnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::TokenizeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe { OPERATION.read_and_free_exception_message(out_error) }?;
            Err(MtmdError::TokenizeReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_BITMAPS_ARG_WHEN_NUM_BITMAPS_NONZERO => {
            Err(OPERATION.contract_violated("nonzero bitmap count was observed with a null bitmap array")
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG => Err(OPERATION.contract_violated("was given a null ctx argument")
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG => Err(OPERATION.contract_violated("was given a null output argument")
        .into()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG => Err(OPERATION.contract_violated("was given a null text argument")
        .into()),
        other => Err(OPERATION.unknown_status(i64::from(other))
        .into()),
    }
}

fn map_tokenize_from_parts_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_tokenize_from_parts_status,
    undocumented_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    const OPERATION: WrapperOperation = WrapperOperation::new("llama_rs_mtmd_tokenize_from_parts");

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART => {
            Err(OPERATION.contract_violated("llama.cpp reported a malformed part list: a part holding both or neither of text and bitmap, a text part with a null text pointer, or bitmaps that do not line up with their parts")
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE => {
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE => {
            Err(MtmdError::TokenizeFromPartsUnknownStatus {
                code: undocumented_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::TokenizeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                OPERATION.read_and_free_exception_message(out_error)
            }?;
            Err(MtmdError::TokenizeReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_CTX_ARG => {
            Err(OPERATION.contract_violated("was given a null ctx argument")
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_OUTPUT_ARG => {
            Err(OPERATION.contract_violated("was given a null output argument")
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PARTS_ARG_WHEN_NUM_PARTS_NONZERO => {
            Err(OPERATION.contract_violated("nonzero part count was observed with a null part array")
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PART_ELEMENT => {
            Err(OPERATION.contract_violated("was given a null element in the parts argument")
            .into())
        }
        other => Err(OPERATION.unknown_status(i64::from(other))
        .into()),
    }
}

fn fit_each(
    chunks: &[MtmdInputChunk],
    mtmd_ctx: &MtmdContext,
    micro_batch_tokens: u32,
) -> Result<Vec<MtmdInputChunkType>, MtmdError> {
    chunks
        .iter()
        .map(|chunk| chunk.fit_to_micro_batch(mtmd_ctx, micro_batch_tokens))
        .collect()
}

#[derive(Debug)]
pub struct MtmdInputChunks {
    pub chunks: NonNull<llama_cpp_bindings_sys::mtmd_input_chunks>,
}

unsafe impl Send for MtmdInputChunks {}

impl MtmdInputChunks {
    /// # Errors
    ///
    /// Returns the [`MtmdError`] variant matching the wrapper's status code.
    pub fn new() -> Result<Self, MtmdError> {
        let mut out_chunks: *mut llama_cpp_bindings_sys::mtmd_input_chunks = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_input_chunks_init(
                &raw mut out_chunks,
                &raw mut out_error,
            )
        };

        input_chunks_init_status_to_result(status, out_chunks, out_error)
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] variant matching the wrapper's status code.
    pub fn tokenize(
        MtmdContext { context, .. }: &MtmdContext,
        MtmdInputText {
            text,
            add_special,
            parse_special,
        }: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<Self, MtmdError> {
        let chunks = Self::new()
            .map_err(Box::new)
            .map_err(MtmdError::TokenizeInputChunks)?;
        let text_cstring = CString::new(text).map_err(MtmdError::InputTextContainsNul)?;
        let input_text = llama_cpp_bindings_sys::mtmd_input_text {
            text: text_cstring.as_ptr(),
            text_len: text_cstring.as_bytes().len(),
            add_special,
            parse_special,
        };

        let bitmap_ptrs: Vec<*const llama_cpp_bindings_sys::mtmd_bitmap> = bitmaps
            .iter()
            .map(|MtmdBitmap { bitmap, .. }| bitmap.as_ptr().cast_const())
            .collect();

        let mut out_undocumented_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_tokenize(
                context.as_ptr(),
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
    /// Returns an [`MtmdError`] variant matching the wrapper's status code.
    pub fn tokenize_from_parts(
        MtmdContext { context, .. }: &MtmdContext,
        parts: &[MtmdInputPart<'_>],
        add_special: bool,
    ) -> Result<Self, MtmdError> {
        let chunks = Self::new()
            .map_err(Box::new)
            .map_err(MtmdError::TokenizeInputChunks)?;
        let ffi_input_parts = parts
            .iter()
            .map(MtmdFfiInputPart::try_from)
            .collect::<Result<Vec<MtmdFfiInputPart<'_>>, _>>()
            .map_err(MtmdError::InputTextContainsNul)?;
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
                context.as_ptr(),
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

    /// Checks every chunk with [`Self::fit_to_micro_batch`] before evaluating any of them, so a
    /// chunk that cannot be decoded leaves the KV cache untouched.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdEvalLimits::batch_tokens`] or [`Self::fit_to_micro_batch`], or
    /// [`MtmdError::EvalFailed`] if any encoding or decoding operation fails.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdError> {
        let limits = MtmdEvalLimits::of(llama_ctx);
        let micro_batch_tokens = limits
            .context
            .micro_batch_tokens(limits.batch_tokens(n_batch, seq_id)?);
        let chunks = self.chunks_within_count()?;

        fit_each(&chunks, mtmd_ctx, micro_batch_tokens)?;

        let last_chunk_index = chunks.len().checked_sub(1);

        chunks
            .iter()
            .enumerate()
            .try_fold(start_position, |position, (index, chunk)| {
                chunk.eval(
                    mtmd_ctx,
                    llama_ctx,
                    position,
                    seq_id,
                    n_batch,
                    logits_last && Some(index) == last_chunk_index,
                )
            })
    }

    /// Checks every chunk with [`MtmdInputChunk::fit_to_micro_batch`] before any of them is
    /// evaluated, and returns the chunk types it classified, in chunk order.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::NonCausalChunkExceedsMicroBatch`] for the first chunk decoded
    /// non-causally that has more tokens than `micro_batch_tokens`,
    /// [`MtmdError::EvalUnknownChunkType`] when a chunk type is unknown, or
    /// [`MtmdError::Ffi`] when a chunk within the chunk count is null.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<Vec<MtmdInputChunkType>, MtmdError> {
        fit_each(&self.chunks_within_count()?, mtmd_ctx, micro_batch_tokens)
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk> {
        if index >= self.len() {
            return None;
        }

        let chunk_ptr = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_input_chunks_get(self.chunks.as_ptr(), index)
        };

        NonNull::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            owned: false,
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    fn chunks_within_count(&self) -> Result<Vec<MtmdInputChunk>, MtmdError> {
        (0..self.len())
            .map(|index| {
                self.get(index).ok_or_else(|| {
                    WrapperOperation::new("mtmd_input_chunks_get")
                        .contract_violated("returned a null chunk within the chunk count")
                        .into()
                })
            })
            .collect()
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_error::FfiError;
    use llama_cpp_ffi_status::wrapper_operation::WrapperOperation;
    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::input_chunks_init_status_to_result;
    use super::map_tokenize_from_parts_status;
    use super::map_tokenize_status;
    use crate::mtmd_input_chunks::MtmdInputChunks;

    use crate::mtmd_error::MtmdError;

    const OPERATION: &str = "llama_rs_mtmd_input_chunks_init";

    fn contract_violation(detail: &'static str) -> MtmdError {
        FfiError::ContractViolated {
            operation: OPERATION,
            detail,
        }
        .into()
    }

    #[test]
    fn new_creates_empty_chunks() {
        let chunks = MtmdInputChunks::new().unwrap();

        assert!(chunks.is_empty());
        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let chunks = MtmdInputChunks::new().unwrap();

        assert!(chunks.get(0).is_none());
        assert!(chunks.get(999).is_none());
    }

    #[test]
    fn every_failure_status_maps_to_its_error() {
        let statuses = [
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_NULL_OUT_CHUNKS_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
            255,
        ];

        assert_eq!(
            statuses.map(|status| {
                input_chunks_init_status_to_result(status, ptr::null_mut(), ptr::null_mut()).err()
            }),
            [
                Some(contract_violation("success status contained null chunks")),
                Some(contract_violation("was given a null out_chunks argument")),
                Some(MtmdError::InputChunksNotEnoughMemory),
                Some(MtmdError::InputChunksLlamaCppOutOfMemory),
                Some(contract_violation(
                    "reported a thrown C++ exception without an error message"
                )),
                Some(MtmdError::Ffi(FfiError::UnknownStatus {
                    operation: OPERATION,
                    code: 255,
                })),
            ]
        );
    }

    #[test]
    fn a_thrown_exception_reports_its_message() {
        let out_error =
            unsafe { llama_cpp_bindings_sys::llama_rs_string_dup(c"allocation threw".as_ptr()) };

        assert_eq!(
            input_chunks_init_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
            .err(),
            Some(MtmdError::InputChunksReported {
                message: "allocation threw".to_owned(),
            })
        );
    }

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
    fn tokenize_from_parts_status_reports_a_media_preprocessing_or_tokenization_failure() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
                0,
                ptr::null_mut(),
            ),
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
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
            Err(MtmdError::TokenizeFromPartsUnknownStatus { code: 42 })
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
            Err(MtmdError::TokenizeNotEnoughMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_reports_llama_cpp_running_out_of_memory() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            ),
            Err(MtmdError::TokenizeLlamaCppOutOfMemory)
        );
    }

    #[test]
    fn tokenize_from_parts_status_maps_a_thrown_exception_with_its_message() {
        let expected_message = unsafe {
            WrapperOperation::new("llama_rs_mtmd_tokenize_from_parts")
                .read_and_free_exception_message(wrapper_allocated_error_message_pointer())
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
                .map(|message| MtmdError::TokenizeReported { message })
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_malformed_part_list_as_a_contract_violation() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_REPORTED_MALFORMED_PART,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "llama.cpp reported a malformed part list: a part holding both or neither of text and bitmap, a text part with a null text pointer, or bitmaps that do not line up with their parts",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_null_part_element_as_a_contract_violation() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PART_ELEMENT,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null element in the parts argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_an_exception_without_a_message_as_a_contract_violation()
     {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "reported a thrown C++ exception without an error message",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_null_ctx_arg_as_a_contract_violation() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_CTX_ARG,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_null_output_arg_as_a_contract_violation() {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_OUTPUT_ARG,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "was given a null output argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_reports_a_null_parts_arg_with_a_nonzero_part_count_as_a_contract_violation()
     {
        assert_eq!(
            map_tokenize_from_parts_status(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_FROM_PARTS_NULL_PARTS_ARG_WHEN_NUM_PARTS_NONZERO,
                0,
                ptr::null_mut(),
            )
            .err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize_from_parts",
                    detail: "nonzero part count was observed with a null part array",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_from_parts_status_preserves_an_unknown_status() {
        assert_eq!(
            map_tokenize_from_parts_status(255, 0, ptr::null_mut()).err(),
            Some(MtmdError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_mtmd_tokenize_from_parts",
                code: 255,
            }))
        );
    }

    #[test]
    fn tokenize_status_reports_a_bitmap_count_that_does_not_match_the_marker_count() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_BITMAP_COUNT_DOES_NOT_MATCH_MARKER_COUNT,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdError::BitmapCountDoesNotMatchMarkerCount));
    }

    #[test]
    fn tokenize_status_reports_a_media_preprocessing_or_tokenization_failure() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_REPORTED_PREPROCESSING_OR_TOKENIZATION_FAILURE,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(MtmdError::MediaPreprocessingOrTokenizationFailed)
        );
    }

    #[test]
    fn tokenize_status_maps_an_undocumented_code_with_its_value() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_RETURNED_UNDOCUMENTED_NONZERO_CODE,
            42,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdError::TokenizeUnknownStatus { code: 42 }));
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

        assert_eq!(result, Err(MtmdError::TokenizeNotEnoughMemory));
    }

    #[test]
    fn tokenize_status_reports_an_exception_without_a_message_as_a_contract_violation() {
        let result = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
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
            Err(MtmdError::Ffi(FfiError::ContractViolated {
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
            Err(MtmdError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_mtmd_tokenize",
                code: 255,
            }))
        );
    }

    #[test]
    fn map_tokenize_status_reports_a_null_ctx_arg_as_a_contract_violation() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_reports_a_null_output_arg_as_a_contract_violation() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_OUTPUT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null output argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_reports_a_null_text_arg_as_a_contract_violation() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_NULL_TEXT_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_tokenize",
                    detail: "was given a null text argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_tokenize_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = map_tokenize_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_TOKENIZE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdError::TokenizeLlamaCppOutOfMemory));
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
            Err(MtmdError::TokenizeReported {
                message: "tokenizer threw".to_owned(),
            })
        );
    }
}
