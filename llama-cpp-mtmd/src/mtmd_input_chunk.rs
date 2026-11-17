use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::str::Utf8Error;

use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_ffi_status::ffi_error::FfiError;
use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;
use llama_cpp_model::llama_token::LlamaToken;

use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;
use crate::optional_mtmd_id::optional_mtmd_id;

/// # Safety
///
/// `tokens_ptr` must point to at least `n_tokens` valid `llama_token` values
/// that remain valid for the lifetime `'chunk`.
const unsafe fn tokens_from_raw_ptr<'chunk>(
    tokens_ptr: *const llama_cpp_bindings_sys::llama_token,
    n_tokens: usize,
) -> Option<&'chunk [LlamaToken]> {
    if tokens_ptr.is_null() || n_tokens == 0 {
        None
    } else {
        unsafe {
            Some(slice::from_raw_parts(
                tokens_ptr.cast::<LlamaToken>(),
                n_tokens,
            ))
        }
    }
}

fn non_causal_chunk_micro_batch_error(
    decodes_non_causally: bool,
    chunk_tokens: usize,
    micro_batch_tokens: u32,
) -> Option<MtmdError> {
    if decodes_non_causally
        && u64::try_from(chunk_tokens).is_ok_and(|tokens| tokens > u64::from(micro_batch_tokens))
    {
        return Some(MtmdError::NonCausalChunkExceedsMicroBatch(
            NonCausalChunkMicroBatchMismatch {
                chunk_tokens,
                micro_batch_tokens,
            },
        ));
    }

    None
}

fn input_chunk_copy_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_input_chunk_copy_status,
    out_chunk: *mut llama_cpp_bindings_sys::mtmd_input_chunk,
    out_error: *mut c_char,
) -> Result<MtmdInputChunk, MtmdError> {
    const OPERATION: &str = "llama_rs_mtmd_input_chunk_copy";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK => NonNull::new(out_chunk)
            .map(|chunk| MtmdInputChunk { chunk, owned: true })
            .ok_or_else(|| {
                FfiError::ContractViolated {
                    operation: OPERATION,
                    detail: "success status contained a null chunk",
                }
                .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_CHUNK_ARG => {
            Err(FfiError::ContractViolated {
                operation: OPERATION,
                detail: "was given a null chunk argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_OUT_CHUNK_ARG => {
            Err(FfiError::ContractViolated {
                operation: OPERATION,
                detail: "was given a null out_chunk argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::ChunkCopyNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::ChunkCopyLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    OPERATION,
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(MtmdError::ChunkCopyReported { message })
        }
        other => Err(FfiError::UnknownStatus {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

fn map_encode_chunk_status(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk_status,
    llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_OK => Ok(()),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            Err(MtmdError::EncodingFailed {
                code: llama_cpp_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::EncodeNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::EncodeLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_encode_chunk",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdError::EncodeReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "was given a null ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_encode_chunk",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_mtmd_encode_chunk",
            code: i64::from(other),
        }
        .into()),
    }
}

fn eval_chunk_single_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single_status,
    final_position: llama_cpp_bindings_sys::llama_pos,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK => Ok(final_position),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            Err(MtmdError::EvalFailed {
                code: out_llama_cpp_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::EvalNotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::EvalLlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_eval_chunk_single",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdError::EvalReported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null mtmd_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null llama_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG => {
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null out_new_n_past argument",
            }
            .into())
        }
        other => Err(FfiError::UnknownStatus {
            operation: "llama_rs_mtmd_eval_chunk_single",
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Debug)]
pub struct MtmdInputChunk {
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdError> {
        let chunk_type = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_input_chunk_get_type(self.chunk.as_ptr())
        };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// # Errors
    ///
    /// Returns the [`MtmdError`] variant matching the wrapper's status code.
    pub fn copy(&self) -> Result<Self, MtmdError> {
        let mut out_chunk: *mut llama_cpp_bindings_sys::mtmd_input_chunk = ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_input_chunk_copy(
                self.chunk.as_ptr(),
                &raw mut out_chunk,
                &raw mut out_error,
            )
        };

        input_chunk_copy_status_to_result(status, out_chunk, out_error)
    }

    #[must_use]
    pub fn decode_use_non_causal(&self, MtmdContext { context, .. }: &MtmdContext) -> bool {
        unsafe {
            llama_cpp_bindings_sys::mtmd_decode_use_non_causal(
                context.as_ptr(),
                self.chunk.as_ptr(),
            )
        }
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] variant matching the wrapper's status code.
    pub fn encode(&self, MtmdContext { context, .. }: &MtmdContext) -> Result<(), MtmdError> {
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_encode_chunk(
                context.as_ptr(),
                self.chunk.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        map_encode_chunk_status(status, out_llama_cpp_return_code, out_error)
    }

    /// Encodes the chunk when it holds media and decodes it into `llama_ctx` from `start_position`,
    /// returning the position after it.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdError`] variant matching the wrapper's status code.
    pub fn eval(
        &self,
        MtmdContext {
            context: mtmd_context,
            ..
        }: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdError> {
        let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single(
                mtmd_context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunk.as_ptr(),
                start_position,
                seq_id,
                n_batch,
                logits_last,
                &raw mut final_position,
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        let next_position = eval_chunk_single_status_to_result(
            status,
            final_position,
            out_llama_cpp_return_code,
            out_error,
        )?;
        llama_ctx.record_helper_decode(self.n_tokens() as u64, logits_last);

        Ok(next_position)
    }

    /// Checks that this chunk can be evaluated in decodes of `micro_batch_tokens` and returns the
    /// chunk type it classified. llama.cpp splits a causal chunk across decodes, while a media
    /// chunk it decodes non-causally has to fit a single decode.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::NonCausalChunkExceedsMicroBatch`] when a media chunk decoded
    /// non-causally has more tokens than `micro_batch_tokens`, or
    /// [`MtmdError::UnknownInputChunkType`] when the chunk type is unknown.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<MtmdInputChunkType, MtmdError> {
        let chunk_type = self.chunk_type()?;
        let decodes_non_causally =
            chunk_type != MtmdInputChunkType::Text && self.decode_use_non_causal(mtmd_ctx);

        non_causal_chunk_micro_batch_error(
            decodes_non_causally,
            self.n_tokens(),
            micro_batch_tokens,
        )
        .map_or(Ok(chunk_type), Err)
    }

    /// # Errors
    ///
    /// Returns [`Utf8Error`] when the id is not valid UTF-8.
    pub fn id(&self) -> Result<Option<String>, Utf8Error> {
        unsafe {
            optional_mtmd_id(llama_cpp_bindings_sys::mtmd_input_chunk_get_id(
                self.chunk.as_ptr(),
            ))
        }
    }

    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn text_tokens_of(&self, chunk_type: MtmdInputChunkType) -> Option<&[LlamaToken]> {
        if chunk_type != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        unsafe { tokens_from_raw_ptr(tokens_ptr, n_tokens) }
    }
}

impl Drop for MtmdInputChunk {
    fn drop(&mut self) {
        if self.owned {
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_ffi_status::ffi_error::FfiError;
    use llama_cpp_ffi_status::read_and_free_cpp_string::read_and_free_cpp_string;
    use llama_cpp_model::llama_token::LlamaToken;
    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer::wrapper_allocated_error_message_pointer;

    use super::eval_chunk_single_status_to_result;
    use super::input_chunk_copy_status_to_result;
    use super::map_encode_chunk_status;
    use super::non_causal_chunk_micro_batch_error;
    use super::tokens_from_raw_ptr;
    use crate::mtmd_error::MtmdError;
    use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

    const OPERATION: &str = "llama_rs_mtmd_input_chunk_copy";

    fn contract_violation(detail: &'static str) -> MtmdError {
        FfiError::ContractViolated {
            operation: OPERATION,
            detail,
        }
        .into()
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert_eq!(unsafe { tokens_from_raw_ptr(ptr::null(), 5) }, None);
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::llama_token = 42;

        assert_eq!(unsafe { tokens_from_raw_ptr(&raw const token, 0) }, None);
    }

    #[test]
    fn tokens_from_raw_ptr_views_every_token() {
        let tokens: [llama_cpp_bindings_sys::llama_token; 2] = [1, 2];
        let result = unsafe { tokens_from_raw_ptr(tokens.as_ptr(), 2) };

        assert_eq!(result, Some(&[LlamaToken::new(1), LlamaToken::new(2)][..]));
    }

    #[test]
    fn a_non_causal_chunk_larger_than_the_micro_batch_reports_the_mismatch() {
        assert_eq!(
            non_causal_chunk_micro_batch_error(true, 9, 4),
            Some(MtmdError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: 9,
                    micro_batch_tokens: 4,
                }
            ))
        );
    }

    #[test]
    fn a_causal_chunk_larger_than_the_micro_batch_is_split_instead() {
        assert_eq!(non_causal_chunk_micro_batch_error(false, 9, 4), None);
    }

    #[test]
    fn a_non_causal_chunk_within_the_micro_batch_fits() {
        assert_eq!(non_causal_chunk_micro_batch_error(true, 4, 4), None);
    }

    #[test]
    fn every_failure_status_maps_to_its_error() {
        let statuses = [
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_CHUNK_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_NULL_OUT_CHUNK_ARG,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY,
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
            255,
        ];

        assert_eq!(
            statuses.map(|status| {
                input_chunk_copy_status_to_result(status, ptr::null_mut(), ptr::null_mut()).err()
            }),
            [
                Some(contract_violation("success status contained a null chunk")),
                Some(contract_violation("was given a null chunk argument")),
                Some(contract_violation("was given a null out_chunk argument")),
                Some(MtmdError::ChunkCopyNotEnoughMemory),
                Some(MtmdError::ChunkCopyLlamaCppOutOfMemory),
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
            input_chunk_copy_status_to_result(
                llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                ptr::null_mut(),
                out_error,
            )
            .err(),
            Some(MtmdError::ChunkCopyReported {
                message: "allocation threw".to_owned(),
            })
        );
    }

    #[test]
    fn eval_chunk_single_status_cxx_exception_surfaces_the_reported_message() {
        let expected_message = unsafe {
            read_and_free_cpp_string(
                wrapper_allocated_error_message_pointer(),
                "llama_rs_mtmd_eval_chunk_single",
                "reported a thrown C++ exception without an error message",
            )
        };
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            0,
            wrapper_allocated_error_message_pointer(),
        );

        assert_eq!(
            result.err(),
            expected_message
                .ok()
                .map(|message| MtmdError::EvalReported { message })
        );
    }

    #[test]
    fn eval_chunk_single_status_ok_returns_final_position() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK,
            7,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Ok(7));
    }

    #[test]
    fn eval_chunk_single_status_nonzero_code_maps_to_eval_failed() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            0,
            -3,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdError::EvalFailed { code: -3 }));
    }

    #[test]
    fn eval_chunk_single_status_allocation_failed_maps_to_not_enough_memory() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdError::EvalNotEnoughMemory));
    }

    #[test]
    fn eval_chunk_single_status_reports_an_exception_without_a_message_as_a_contract_violation() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn eval_chunk_single_unknown_status_is_preserved() {
        let result = eval_chunk_single_status_to_result(255, 0, 0, ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_mtmd_eval_chunk_single",
                code: 255,
            }))
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_a_null_mtmd_ctx_arg_as_a_contract_violation() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null mtmd_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_a_null_llama_ctx_arg_as_a_contract_violation() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null llama_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_a_null_chunk_arg_as_a_contract_violation() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_a_null_out_new_n_past_arg_as_a_contract_violation()
     {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null out_new_n_past argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdError::EvalLlamaCppOutOfMemory));
    }

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

        assert_eq!(result, Err(MtmdError::EncodingFailed { code: 5 }));
    }

    #[test]
    fn encode_chunk_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdError::EncodeNotEnoughMemory));
    }

    #[test]
    fn encode_chunk_status_reports_an_exception_without_a_message_as_a_contract_violation() {
        let result = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(FfiError::ContractViolated {
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
            Err(MtmdError::Ffi(FfiError::UnknownStatus {
                operation: "llama_rs_mtmd_encode_chunk",
                code: 255,
            }))
        );
    }

    #[test]
    fn map_encode_chunk_status_reports_a_null_ctx_arg_as_a_contract_violation() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CTX_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_encode_chunk_status_reports_a_null_chunk_arg_as_a_contract_violation() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_NULL_CHUNK_ARG,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                FfiError::ContractViolated {
                    operation: "llama_rs_mtmd_encode_chunk",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn map_encode_chunk_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = map_encode_chunk_status(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdError::EncodeLlamaCppOutOfMemory));
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
            Err(MtmdError::EncodeReported {
                message: "encoder threw".to_owned(),
            })
        );
    }
}
