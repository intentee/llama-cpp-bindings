use std::ffi::CStr;
use std::ffi::c_char;
use std::ptr::NonNull;
use std::slice;

use crate::context::LlamaContext;
use crate::token::LlamaToken;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use super::micro_batch_tokens::micro_batch_tokens;
use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk_error::MtmdInputChunkError;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunk_type_error::MtmdInputChunkTypeError;
use super::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

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

fn eval_chunk_single_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single_status,
    final_position: llama_cpp_bindings_sys::llama_pos,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK => Ok(final_position),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_RETURNED_NONZERO_CODE => {
            Err(MtmdEvalError::EvalFailed {
                code: out_llama_cpp_return_code,
            })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdEvalError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdEvalError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    "llama_rs_mtmd_eval_chunk_single",
                    "reported a thrown C++ exception without an error message",
                )
            }?;
            Err(MtmdEvalError::Reported { message })
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null mtmd_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null llama_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG => {
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null out_new_n_past argument",
            }
            .into())
        }
        other => Err(crate::FfiStatusError {
            operation: "llama_rs_mtmd_eval_chunk_single",
            code: i64::from(other),
        }
        .into()),
    }
}

fn non_causal_chunk_micro_batch_error(
    decodes_non_causally: bool,
    chunk_tokens: usize,
    micro_batch_tokens: u32,
) -> Option<MtmdEvalError> {
    if decodes_non_causally
        && u64::try_from(chunk_tokens).is_ok_and(|tokens| tokens > u64::from(micro_batch_tokens))
    {
        return Some(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
            NonCausalChunkMicroBatchMismatch {
                chunk_tokens,
                micro_batch_tokens,
            },
        ));
    }

    None
}

#[derive(Debug)]
pub struct MtmdInputChunk {
    pub chunk: NonNull<llama_cpp_bindings_sys::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdInputChunkTypeError> {
        let chunk_type =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// # Errors
    ///
    /// Returns [`MtmdInputChunkTypeError`] when the wrapper reports a chunk type this
    /// binding does not know, so an unclassifiable chunk is never mistaken for a
    /// non-text chunk.
    pub fn text_tokens(&self) -> Result<Option<&[LlamaToken]>, MtmdInputChunkTypeError> {
        if self.chunk_type()? != MtmdInputChunkType::Text {
            return Ok(None);
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        Ok(unsafe { tokens_from_raw_ptr(tokens_ptr, n_tokens) })
    }

    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn id(&self) -> Option<String> {
        let ptr = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
                .into()
        }
    }

    /// # Errors
    ///
    /// Returns `MtmdInputChunkError::ChunkOperationFailed` if copying fails.
    pub fn copy(&self) -> Result<Self, MtmdInputChunkError> {
        let chunk = unsafe { llama_cpp_bindings_sys::mtmd_input_chunk_copy(self.chunk.as_ptr()) };
        let chunk = NonNull::new(chunk).ok_or(MtmdInputChunkError::ChunkOperationFailed)?;

        Ok(Self { chunk, owned: true })
    }

    /// Checks that this chunk can be evaluated in decodes of `micro_batch_tokens`. llama.cpp
    /// splits a causal chunk across decodes, while a media chunk it decodes non-causally has to
    /// fit a single decode.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonCausalChunkExceedsMicroBatch`] when a media chunk decoded
    /// non-causally has more tokens than `micro_batch_tokens`, or
    /// [`MtmdEvalError::UnknownChunkType`] when the chunk type is unknown.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<(), MtmdEvalError> {
        let decodes_non_causally =
            self.chunk_type()? != MtmdInputChunkType::Text && mtmd_ctx.decode_use_non_causal(self);

        non_causal_chunk_micro_batch_error(
            decodes_non_causally,
            self.n_tokens(),
            micro_batch_tokens,
        )
        .map_or(Ok(()), Err)
    }

    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonPositiveBatchSize`] when `n_batch` is not positive,
    /// [`MtmdEvalError::NonCausalChunkExceedsMicroBatch`] when this chunk has to fit a single
    /// decode but does not, or [`MtmdEvalError::EvalFailed`] if the underlying encode or decode
    /// step fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        self.fit_to_micro_batch(mtmd_ctx, micro_batch_tokens(llama_ctx, n_batch)?)?;

        let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single(
                mtmd_ctx.context.as_ptr(),
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

        eval_chunk_single_status_to_result(
            status,
            final_position,
            out_llama_cpp_return_code,
            out_error,
        )
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
mod unit_tests {
    use super::eval_chunk_single_status_to_result;
    use super::non_causal_chunk_micro_batch_error;
    use super::tokens_from_raw_ptr;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;
    use crate::mtmd::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert!(unsafe { tokens_from_raw_ptr(std::ptr::null(), 5) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::llama_token = 42;
        assert!(unsafe { tokens_from_raw_ptr(&raw const token, 0) }.is_none());
    }

    #[test]
    fn tokens_from_raw_ptr_returns_some_for_valid() {
        let tokens: [llama_cpp_bindings_sys::llama_token; 2] = [1, 2];
        let result = unsafe { tokens_from_raw_ptr(tokens.as_ptr(), 2) };

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
    }

    #[test]
    fn eval_chunk_single_status_ok_returns_final_position() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK,
            7,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Ok(7));
    }

    #[test]
    fn eval_chunk_single_status_nonzero_code_maps_to_eval_failed() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_RETURNED_NONZERO_CODE,
            0,
            -3,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEvalError::EvalFailed { code: -3 }));
    }

    #[test]
    fn eval_chunk_single_status_allocation_failed_maps_to_not_enough_memory() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEvalError::NotEnoughMemory));
    }

    #[test]
    fn eval_chunk_single_status_cxx_exception_reports_unknown_error_for_null() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            0,
            std::ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(crate::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "reported a thrown C++ exception without an error message",
            }
            .into())
        );
    }

    #[test]
    fn eval_chunk_single_unknown_status_is_preserved() {
        let result = eval_chunk_single_status_to_result(255, 0, 0, std::ptr::null_mut());

        assert_eq!(
            result,
            Err(MtmdEvalError::FfiStatus(crate::FfiStatusError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                code: 255,
            }))
        );
    }

    #[test]
    fn a_non_causal_chunk_larger_than_the_micro_batch_reports_the_mismatch() {
        assert_eq!(
            non_causal_chunk_micro_batch_error(true, 9, 4),
            Some(MtmdEvalError::NonCausalChunkExceedsMicroBatch(
                NonCausalChunkMicroBatchMismatch {
                    chunk_tokens: 9,
                    micro_batch_tokens: 4,
                }
            ))
        );
    }

    #[test]
    fn a_causal_chunk_larger_than_the_micro_batch_is_split_instead() {
        assert!(non_causal_chunk_micro_batch_error(false, 9, 4).is_none());
    }

    #[test]
    fn a_non_causal_chunk_within_the_micro_batch_fits() {
        assert!(non_causal_chunk_micro_batch_error(true, 4, 4).is_none());
    }
}

#[cfg(test)]
mod ffi_contract_status_tests {
    use super::eval_chunk_single_status_to_result;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;
    use std::ptr;

    #[test]
    fn eval_chunk_single_status_to_result_maps_every_contract_status() {
        let outcome_0 = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_0.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null mtmd_ctx argument",
                }
                .into()
            )
        );
        let outcome_1 = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_1.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null llama_ctx argument",
                }
                .into()
            )
        );
        let outcome_2 = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_2.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
        let outcome_3 = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome_3.err(),
            Some(
                crate::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null out_new_n_past argument",
                }
                .into()
            )
        );
        let outcome_4 = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome_4.err(), Some(MtmdEvalError::LlamaCppOutOfMemory));
    }
}
