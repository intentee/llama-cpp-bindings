use std::ffi::c_char;
use std::ptr;
use std::ptr::NonNull;

use llama_cpp_ffi_status::read_and_free_cpp_string;

use crate::context::LlamaContext;

use super::eval_chunk::eval_chunk;
use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_eval_limits::MtmdEvalLimits;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunks_error::MtmdInputChunksError;

fn input_chunks_init_status_to_result(
    status: llama_cpp_bindings_sys::llama_rs_mtmd_input_chunks_init_status,
    out_chunks: *mut llama_cpp_bindings_sys::mtmd_input_chunks,
    out_error: *mut c_char,
) -> Result<MtmdInputChunks, MtmdInputChunksError> {
    const OPERATION: &str = "llama_rs_mtmd_input_chunks_init";

    match status {
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_OK => NonNull::new(out_chunks)
            .map(|chunks| MtmdInputChunks { chunks })
            .ok_or_else(|| {
                crate::FfiContractError {
                    operation: OPERATION,
                    detail: "success status contained null chunks",
                }
                .into()
            }),
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_NULL_OUT_CHUNKS_ARG => {
            Err(crate::FfiContractError {
                operation: OPERATION,
                detail: "was given a null out_chunks argument",
            }
            .into())
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_ERROR_STRING_ALLOCATION_FAILED => {
            Err(MtmdInputChunksError::NotEnoughMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_OUT_OF_MEMORY => {
            Err(MtmdInputChunksError::LlamaCppOutOfMemory)
        }
        llama_cpp_bindings_sys::LLAMA_RS_MTMD_INPUT_CHUNKS_INIT_LLAMA_CPP_THREW_CXX_EXCEPTION => {
            let message = unsafe {
                read_and_free_cpp_string(
                    out_error,
                    OPERATION,
                    "reported a thrown C++ exception without an error message",
                )
            }?;

            Err(MtmdInputChunksError::Reported { message })
        }
        other => Err(crate::FfiStatusError {
            operation: OPERATION,
            code: i64::from(other),
        }
        .into()),
    }
}

#[derive(Debug)]
pub struct MtmdInputChunks {
    pub chunks: NonNull<llama_cpp_bindings_sys::mtmd_input_chunks>,
}

unsafe impl Send for MtmdInputChunks {}

impl MtmdInputChunks {
    /// # Errors
    ///
    /// Returns the [`MtmdInputChunksError`] variant matching the wrapper's status code.
    pub fn new() -> Result<Self, MtmdInputChunksError> {
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

    #[must_use]
    pub fn len(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    /// Checks every chunk with [`MtmdInputChunk::fit_to_micro_batch`] before any of them is
    /// evaluated, and returns the chunk types it classified, in chunk order.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdEvalError::NonCausalChunkExceedsMicroBatch`] for the first chunk decoded
    /// non-causally that has more tokens than `micro_batch_tokens`,
    /// [`MtmdEvalError::UnknownChunkType`] when a chunk type is unknown, or
    /// [`MtmdEvalError::FfiContract`] when a chunk within the chunk count is null.
    pub fn fit_to_micro_batch(
        &self,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<Vec<MtmdInputChunkType>, MtmdEvalError> {
        (0..self.len())
            .map(|index| {
                self.chunk_within_count(index)?
                    .fit_to_micro_batch(mtmd_ctx, micro_batch_tokens)
            })
            .collect()
    }

    #[must_use]
    pub fn total_tokens(&self) -> usize {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_tokens(self.chunks.as_ptr()) }
    }

    #[must_use]
    pub fn total_positions(&self) -> i32 {
        unsafe { llama_cpp_bindings_sys::mtmd_helper_get_n_pos(self.chunks.as_ptr()) }
    }

    /// Checks every chunk with [`Self::fit_to_micro_batch`] before evaluating any of them, so a
    /// chunk that cannot be decoded leaves the KV cache untouched.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdEvalLimits::batch_tokens`] or [`Self::fit_to_micro_batch`], or
    /// [`MtmdEvalError::EvalFailed`] if any encoding or decoding operation fails.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        let limits = MtmdEvalLimits::of(llama_ctx);

        self.fit_to_micro_batch(
            mtmd_ctx,
            limits
                .context
                .micro_batch_tokens(limits.batch_tokens(n_batch, seq_id)?),
        )?;

        let chunk_count = self.len();

        (0..chunk_count).try_fold(start_position, |position, index| {
            eval_chunk(
                &self.chunk_within_count(index)?,
                mtmd_ctx,
                llama_ctx,
                position,
                seq_id,
                n_batch,
                logits_last && index + 1 == chunk_count,
            )
        })
    }

    fn chunk_within_count(&self, index: usize) -> Result<MtmdInputChunk, MtmdEvalError> {
        self.get(index).ok_or_else(|| {
            crate::FfiContractError {
                operation: "mtmd_input_chunks_get",
                detail: "returned a null chunk within the chunk count",
            }
            .into()
        })
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

#[cfg(test)]
mod tests {
    use super::MtmdInputChunks;

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
}

#[cfg(test)]
mod input_chunks_init_status_to_result_tests {
    use std::ptr;

    use super::input_chunks_init_status_to_result;
    use crate::mtmd::mtmd_input_chunks_error::MtmdInputChunksError;

    const OPERATION: &str = "llama_rs_mtmd_input_chunks_init";

    fn contract_violation(detail: &'static str) -> MtmdInputChunksError {
        crate::FfiContractError {
            operation: OPERATION,
            detail,
        }
        .into()
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
                Some(MtmdInputChunksError::NotEnoughMemory),
                Some(MtmdInputChunksError::LlamaCppOutOfMemory),
                Some(contract_violation(
                    "reported a thrown C++ exception without an error message"
                )),
                Some(MtmdInputChunksError::FfiStatus(crate::FfiStatusError {
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
            Some(MtmdInputChunksError::Reported {
                message: "allocation threw".to_owned(),
            })
        );
    }
}
