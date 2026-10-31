use std::ffi::c_char;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::context::LlamaContext;
use crate::context::decoded_output_rows::DecodedOutputRows;

use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk::MtmdInputChunk;

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
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null mtmd_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null llama_ctx argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null chunk argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG => {
            Err(llama_cpp_ffi_status::FfiContractError {
                operation: "llama_rs_mtmd_eval_chunk_single",
                detail: "was given a null out_new_n_past argument",
            }
            .into())
        }
        other => Err(llama_cpp_ffi_status::FfiStatusError {
            operation: "llama_rs_mtmd_eval_chunk_single",
            code: i64::from(other),
        }
        .into()),
    }
}

/// Encodes `chunk` when it holds media and decodes it into `llama_ctx` from `start_position`,
/// returning the position after it.
pub fn eval_chunk(
    chunk: &MtmdInputChunk,
    mtmd_ctx: &MtmdContext,
    llama_ctx: &mut LlamaContext,
    start_position: llama_cpp_bindings_sys::llama_pos,
    seq_id: llama_cpp_bindings_sys::llama_seq_id,
    n_batch: i32,
    logits_last: bool,
) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
    let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;
    let mut out_llama_cpp_return_code: i32 = 0;
    let mut out_error: *mut c_char = std::ptr::null_mut();

    let status = unsafe {
        llama_cpp_bindings_sys::llama_rs_mtmd_eval_chunk_single(
            mtmd_ctx.context.as_ptr(),
            llama_ctx.context.as_ptr(),
            chunk.chunk.as_ptr(),
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
    llama_ctx.record_decoded_rows(DecodedOutputRows::of_helper_decode(logits_last));

    Ok(next_position)
}

#[cfg(test)]
mod eval_chunk_single_status_to_result_tests {
    use std::ptr;

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

        assert_eq!(result, Err(MtmdEvalError::EvalFailed { code: -3 }));
    }

    #[test]
    fn eval_chunk_single_status_allocation_failed_maps_to_not_enough_memory() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
            0,
            0,
            ptr::null_mut(),
        );

        assert_eq!(result, Err(MtmdEvalError::NotEnoughMemory));
    }

    #[test]
    fn eval_chunk_single_status_cxx_exception_reports_unknown_error_for_null() {
        let result = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
            0,
            0,
            ptr::null_mut(),
        );

        assert_eq!(
            result,
            Err(llama_cpp_ffi_status::FfiContractError {
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
            Err(MtmdEvalError::FfiStatus(
                llama_cpp_ffi_status::FfiStatusError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    code: 255,
                }
            ))
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_maps_null_mtmd_ctx_arg() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_MTMD_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null mtmd_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_maps_null_llama_ctx_arg() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_LLAMA_CTX_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null llama_ctx argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_maps_null_chunk_arg() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_CHUNK_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null chunk argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_maps_null_out_new_n_past_arg() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_NULL_OUT_NEW_N_PAST_ARG,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(
            outcome.err(),
            Some(
                llama_cpp_ffi_status::FfiContractError {
                    operation: "llama_rs_mtmd_eval_chunk_single",
                    detail: "was given a null out_new_n_past argument",
                }
                .into()
            )
        );
    }

    #[test]
    fn eval_chunk_single_status_to_result_maps_llama_cpp_out_of_memory() {
        let outcome = eval_chunk_single_status_to_result(
            llama_cpp_bindings_sys::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
            0,
            0,
            ptr::null_mut(),
        );
        assert_eq!(outcome.err(), Some(MtmdEvalError::LlamaCppOutOfMemory));
    }
}
