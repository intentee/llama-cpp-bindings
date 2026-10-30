use std::ptr::NonNull;

use crate::context::LlamaContext;
use crate::context::decoded_output_rows::DecodedOutputRows;

use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_eval_limits::MtmdEvalLimits;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunks_error::MtmdInputChunksError;

const fn check_eval_result(result: i32) -> Result<(), MtmdEvalError> {
    if result == 0 {
        Ok(())
    } else {
        Err(MtmdEvalError::EvalFailed { code: result })
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
    /// Returns `MtmdInputChunksError::ChunksCreationFailed` if the underlying llama.cpp function
    /// returns null.
    ///
    pub fn new() -> Result<Self, MtmdInputChunksError> {
        let chunks = unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_init() };
        let chunks = NonNull::new(chunks).ok_or(MtmdInputChunksError::ChunksCreationFailed)?;

        Ok(Self { chunks })
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

        let chunk_ptr =
            unsafe { llama_cpp_bindings_sys::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

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
                self.get(index)
                    .ok_or(crate::FfiContractError {
                        operation: "mtmd_input_chunks_get",
                        detail: "returned a null chunk within the chunk count",
                    })?
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

        let mut final_position: llama_cpp_bindings_sys::llama_pos = start_position;

        let result = unsafe {
            llama_cpp_bindings_sys::mtmd_helper_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                start_position,
                seq_id,
                n_batch,
                logits_last,
                &raw mut final_position,
            )
        };

        check_eval_result(result)?;
        llama_ctx.record_decoded_rows(DecodedOutputRows::of_helper_decode(logits_last));

        Ok(final_position)
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

    #[test]
    fn check_eval_result_ok_for_zero() {
        use super::check_eval_result;

        assert!(check_eval_result(0).is_ok());
    }

    #[test]
    fn check_eval_result_error_for_nonzero() {
        use super::MtmdEvalError;
        use super::check_eval_result;

        let err = check_eval_result(7).unwrap_err();
        let representative = MtmdEvalError::EvalFailed { code: 0 };

        assert_eq!(
            std::mem::discriminant(&err),
            std::mem::discriminant(&representative)
        );
    }
}
