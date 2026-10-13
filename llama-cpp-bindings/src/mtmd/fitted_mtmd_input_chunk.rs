use std::borrow::Borrow;
use std::ffi::c_char;
use std::marker::PhantomData;

use crate::context::LlamaContext;
use llama_cpp_ffi_status::read_and_free_cpp_string;

use super::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use super::micro_batch_tokens::micro_batch_tokens;
use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;
use super::positive_batch_tokens::positive_batch_tokens;

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

#[derive(Debug)]
pub struct FittedMtmdInputChunk<'fitted> {
    chunk: MtmdInputChunk,
    fitted_chunks: PhantomData<&'fitted MtmdInputChunks>,
    micro_batch_tokens: u32,
}

impl<'fitted> FittedMtmdInputChunk<'fitted> {
    #[must_use]
    pub fn at<TChunks>(
        fitted_chunks: &'fitted FittedMtmdInputChunks<TChunks>,
        chunk_index: usize,
    ) -> Option<Self>
    where
        TChunks: Borrow<MtmdInputChunks>,
    {
        fitted_chunks.get(chunk_index).map(|chunk| Self {
            chunk,
            fitted_chunks: PhantomData,
            micro_batch_tokens: fitted_chunks.micro_batch_tokens(),
        })
    }

    #[must_use]
    pub const fn chunk(&self) -> &MtmdInputChunk {
        &self.chunk
    }

    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonPositiveBatchSize`] when `n_batch` is not positive,
    /// [`MtmdEvalError::EvaluatedMicroBatchSmallerThanFitted`] when decodes of `n_batch` in
    /// `llama_ctx` are smaller than the micro batch the chunks were fitted to, or
    /// [`MtmdEvalError::EvalFailed`] if the underlying encode or decode step fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        let evaluated_micro_batch_tokens =
            micro_batch_tokens(llama_ctx, positive_batch_tokens(n_batch)?);

        if evaluated_micro_batch_tokens < self.micro_batch_tokens {
            return Err(MtmdEvalError::EvaluatedMicroBatchSmallerThanFitted {
                evaluated_micro_batch_tokens,
                fitted_micro_batch_tokens: self.micro_batch_tokens,
            });
        }

        let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = std::ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunk.chunk.as_ptr(),
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

#[cfg(test)]
mod unit_tests {
    use llama_cpp_ffi_status::read_and_free_cpp_string;
    use llama_cpp_wrapper_error_fixture::wrapper_allocated_error_message_pointer;

    use super::eval_chunk_single_status_to_result;
    use crate::mtmd::mtmd_eval_error::MtmdEvalError;

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
                .map(|message| MtmdEvalError::Reported { message })
        );
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
