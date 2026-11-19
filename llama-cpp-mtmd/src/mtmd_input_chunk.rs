use std::ffi::c_char;
use std::num::NonZeroI32;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::str::Utf8Error;

use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_encode_chunk_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_eval_chunk_single_status;
use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy_status;
use llama_cpp_context::llama_context::LlamaContext;
use llama_cpp_model::llama_token::LlamaToken;

use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;
use crate::optional_mtmd_id::optional_mtmd_id;
use crate::read_exception_message::read_exception_message;

/// # Safety
///
/// `tokens_ptr` must point to at least `n_tokens` valid `llama_token` values
/// that remain valid for the lifetime `'chunk`.
const unsafe fn tokens_from_raw_ptr<'chunk>(
    tokens_ptr: *const llama_cpp_bindings_sys::bindings::llama_token,
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

/// # Safety
///
/// On `LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK` `out_chunk` must be the non-null chunk llama.cpp copied,
/// and on `LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; the message is freed here.
unsafe fn input_chunk_copy_status_to_result(
    status: llama_rs_mtmd_input_chunk_copy_status,
    out_chunk: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunk,
    out_error: *mut c_char,
) -> Result<MtmdInputChunk, MtmdError> {
    match status {
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_OK => Ok(MtmdInputChunk {
            chunk: unsafe { NonNull::new_unchecked(out_chunk) },
            owned: true,
        }),
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::ChunkCopyNotEnoughMemory)
        }
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::ChunkCopyLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_exception_message("llama_rs_mtmd_input_chunk_copy", out_error) }?;

            Err(MtmdError::ChunkCopyReported { message })
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the non-null
/// message the C++ side allocated; it is freed here.
unsafe fn map_encode_chunk_status(
    status: llama_rs_mtmd_encode_chunk_status,
    llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<(), MtmdError> {
    match status {
        llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_OK => NonZeroI32::new(llama_cpp_return_code)
            .map_or(Ok(()), |code| Err(MtmdError::EncodingFailed { code: code.get() })),
        llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::EncodeNotEnoughMemory)
        }
        llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::EncodeLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_exception_message("llama_rs_mtmd_encode_chunk", out_error) }?;
            Err(MtmdError::EncodeReported { message })
        }
    }
}

/// # Safety
///
/// On `LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION` `out_error` must be the
/// non-null message the C++ side allocated; it is freed here.
unsafe fn eval_chunk_single_status_to_result(
    status: llama_rs_mtmd_eval_chunk_single_status,
    final_position: llama_cpp_bindings_sys::bindings::llama_pos,
    out_llama_cpp_return_code: i32,
    out_error: *mut c_char,
) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
    match status {
        llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK => NonZeroI32::new(out_llama_cpp_return_code)
            .map_or(Ok(final_position), |code| {
                Err(MtmdError::EvalFailed { code: code.get() })
            }),
        llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdError::EvalNotEnoughMemory)
        }
        llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdError::EvalLlamaCppOutOfMemory)
        }
        llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message =
                unsafe { read_exception_message("llama_rs_mtmd_eval_chunk_single", out_error) }?;
            Err(MtmdError::EvalReported { message })
        }
    }
}

#[derive(Debug)]
pub struct MtmdInputChunk {
    pub chunk: NonNull<llama_cpp_bindings_sys::bindings::mtmd_input_chunk>,
    pub owned: bool,
}

impl MtmdInputChunk {
    /// # Errors
    /// Returns an error if the chunk type is unknown.
    pub fn chunk_type(&self) -> Result<MtmdInputChunkType, MtmdError> {
        let chunk_type = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_get_type(
                self.chunk.as_ptr(),
            )
        };
        MtmdInputChunkType::try_from(chunk_type)
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to copy the chunk.
    pub fn copy(&self) -> Result<Self, MtmdError> {
        let mut out_chunk: *mut llama_cpp_bindings_sys::bindings::mtmd_input_chunk =
            ptr::null_mut();
        let mut out_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy(
                self.chunk.as_ptr(),
                &raw mut out_chunk,
                &raw mut out_error,
            )
        };

        unsafe { input_chunk_copy_status_to_result(status, out_chunk, out_error) }
    }

    #[must_use]
    pub fn decode_use_non_causal(&self, MtmdContext { context, .. }: &MtmdContext) -> bool {
        unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_decode_use_non_causal(
                context.as_ptr(),
                self.chunk.as_ptr(),
            )
        }
    }

    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to encode the chunk.
    pub fn encode(&self, MtmdContext { context, .. }: &MtmdContext) -> Result<(), MtmdError> {
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_encode_chunk(
                context.as_ptr(),
                self.chunk.as_ptr(),
                &raw mut out_llama_cpp_return_code,
                &raw mut out_error,
            )
        };

        unsafe { map_encode_chunk_status(status, out_llama_cpp_return_code, out_error) }
    }

    /// Encodes the chunk when it holds media and decodes it into `llama_ctx` from `start_position`,
    /// returning the position after it.
    ///
    /// # Errors
    ///
    /// Returns an [`MtmdError`] when llama.cpp fails to encode or decode the chunk.
    pub fn eval(
        &self,
        MtmdContext {
            context: mtmd_context,
            ..
        }: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
        let mut final_position: llama_cpp_bindings_sys::bindings::llama_pos = start_position;
        let mut out_llama_cpp_return_code: i32 = 0;
        let mut out_error: *mut c_char = ptr::null_mut();

        let status = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_mtmd_eval_chunk_single(
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

        let next_position = unsafe {
            eval_chunk_single_status_to_result(
                status,
                final_position,
                out_llama_cpp_return_code,
                out_error,
            )
        }?;
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
    /// [`MtmdError::EvalUnknownChunkType`] when the chunk type is unknown.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<MtmdInputChunkType, MtmdError> {
        let chunk_type = self
            .chunk_type()
            .map_err(Box::new)
            .map_err(MtmdError::EvalUnknownChunkType)?;
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
            optional_mtmd_id(llama_cpp_bindings_sys::bindings::mtmd_input_chunk_get_id(
                self.chunk.as_ptr(),
            ))
        }
    }

    #[must_use]
    pub fn n_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::bindings::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) }
    }

    #[must_use]
    pub fn n_tokens(&self) -> usize {
        unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr())
        }
    }

    #[must_use]
    pub fn text_tokens_of(&self, chunk_type: MtmdInputChunkType) -> Option<&[LlamaToken]> {
        if chunk_type != MtmdInputChunkType::Text {
            return None;
        }

        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_bindings_sys::bindings::mtmd_input_chunk_get_tokens_text(
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
            unsafe { llama_cpp_bindings_sys::bindings::mtmd_input_chunk_free(self.chunk.as_ptr()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_encode_chunk_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_eval_chunk_single_status;
    use llama_cpp_bindings_sys::bindings::llama_rs_mtmd_input_chunk_copy_status;
    use llama_cpp_model::llama_token::LlamaToken;

    use super::eval_chunk_single_status_to_result;
    use super::input_chunk_copy_status_to_result;
    use super::map_encode_chunk_status;
    use super::non_causal_chunk_micro_batch_error;
    use super::tokens_from_raw_ptr;
    use crate::mtmd_error::MtmdError;
    use crate::non_causal_chunk_micro_batch_mismatch::NonCausalChunkMicroBatchMismatch;

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_null() {
        assert_eq!(unsafe { tokens_from_raw_ptr(ptr::null(), 5) }, None);
    }

    #[test]
    fn tokens_from_raw_ptr_returns_none_for_zero_count() {
        let token: llama_cpp_bindings_sys::bindings::llama_token = 42;

        assert_eq!(unsafe { tokens_from_raw_ptr(&raw const token, 0) }, None);
    }

    #[test]
    fn tokens_from_raw_ptr_views_every_token() {
        let tokens: [llama_cpp_bindings_sys::bindings::llama_token; 2] = [1, 2];
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
            llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_ERROR_STRING_ALLOCATION_FAILED,
            llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_OUT_OF_MEMORY,
        ];

        assert_eq!(
            statuses.map(|status| {
                unsafe {
                    input_chunk_copy_status_to_result(status, ptr::null_mut(), ptr::null_mut())
                }
                .err()
            }),
            [
                Some(MtmdError::ChunkCopyNotEnoughMemory),
                Some(MtmdError::ChunkCopyLlamaCppOutOfMemory),
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
                input_chunk_copy_status_to_result(
                    llama_rs_mtmd_input_chunk_copy_status::LLAMA_RS_MTMD_INPUT_CHUNK_COPY_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    ptr::null_mut(),
                    out_error,
                )
            }
            .err(),
            Some(MtmdError::ChunkCopyReported {
                message: "allocation threw".to_owned(),
            })
        );
    }

    #[test]
    fn eval_chunk_single_status_cxx_exception_surfaces_the_reported_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"evaluation threw".as_ptr())
        };
        let result = unsafe {
            eval_chunk_single_status_to_result(
                llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_THREW_CXX_EXCEPTION,
                0,
                0,
                out_error,
            )
        };

        assert_eq!(
            result.err(),
            Some(MtmdError::EvalReported {
                message: "evaluation threw".to_owned(),
            })
        );
    }

    #[test]
    fn eval_chunk_single_status_ok_returns_final_position() {
        let result = unsafe {
            eval_chunk_single_status_to_result(
                llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK,
                7,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(7));
    }

    #[test]
    fn eval_chunk_single_nonzero_llama_cpp_code_maps_to_eval_failed() {
        let result = unsafe {
            eval_chunk_single_status_to_result(
                llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_OK,
                0,
                -3,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::EvalFailed { code: -3 }));
    }

    #[test]
    fn eval_chunk_single_status_allocation_failed_maps_to_not_enough_memory() {
        let result = unsafe {
            eval_chunk_single_status_to_result(
                llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_ERROR_STRING_ALLOCATION_FAILED,
                0,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::EvalNotEnoughMemory));
    }

    #[test]
    fn eval_chunk_single_status_to_result_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            eval_chunk_single_status_to_result(
                llama_rs_mtmd_eval_chunk_single_status::LLAMA_RS_MTMD_EVAL_CHUNK_SINGLE_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(MtmdError::EvalLlamaCppOutOfMemory));
    }

    #[test]
    fn encode_chunk_status_maps_ok_to_unit() {
        let result = unsafe {
            map_encode_chunk_status(
                llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_OK,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Ok(()));
    }

    #[test]
    fn encode_chunk_nonzero_llama_cpp_code_maps_to_encoding_failed() {
        let result = unsafe {
            map_encode_chunk_status(
                llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_OK,
                5,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::EncodingFailed { code: 5 }));
    }

    #[test]
    fn encode_chunk_status_maps_string_allocation_failed_to_not_enough_memory() {
        let result = unsafe {
            map_encode_chunk_status(
                llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_ERROR_STRING_ALLOCATION_FAILED,
                0,
                ptr::null_mut(),
            )
        };

        assert_eq!(result, Err(MtmdError::EncodeNotEnoughMemory));
    }

    #[test]
    fn map_encode_chunk_status_reports_llama_cpp_running_out_of_memory() {
        let outcome = unsafe {
            map_encode_chunk_status(
                llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_OUT_OF_MEMORY,
                0,
                ptr::null_mut(),
            )
        };
        assert_eq!(outcome.err(), Some(MtmdError::EncodeLlamaCppOutOfMemory));
    }

    #[test]
    fn map_encode_chunk_status_maps_a_thrown_exception_with_its_message() {
        let out_error = unsafe {
            llama_cpp_bindings_sys::bindings::llama_rs_string_dup(c"encoder threw".as_ptr())
        };

        assert_eq!(
            unsafe {
                map_encode_chunk_status(
                    llama_rs_mtmd_encode_chunk_status::LLAMA_RS_MTMD_ENCODE_CHUNK_LLAMA_CPP_THREW_CXX_EXCEPTION,
                    0,
                    out_error,
                )
            },
            Err(MtmdError::EncodeReported {
                message: "encoder threw".to_owned(),
            })
        );
    }
}
