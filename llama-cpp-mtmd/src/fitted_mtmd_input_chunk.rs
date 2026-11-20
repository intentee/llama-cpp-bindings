use llama_cpp_context::llama_context::LlamaContext;

use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::mtmd_input_chunk_view::MtmdInputChunkView;

#[derive(Debug)]
pub struct FittedMtmdInputChunk<'chunks, 'context> {
    chunk: MtmdInputChunkView<'chunks, 'context>,
    chunk_type: MtmdInputChunkType,
    micro_batch_tokens: u32,
}

impl<'chunks, 'context> FittedMtmdInputChunk<'chunks, 'context> {
    /// Checks `chunk` once with [`MtmdInputChunkView::fit_to_micro_batch`] and keeps the chunk
    /// type it classified, so the chunk can be evaluated and ingested without checking or
    /// classifying it again.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdInputChunkView::fit_to_micro_batch`].
    pub fn fit(
        chunk: MtmdInputChunkView<'chunks, 'context>,
        micro_batch_tokens: u32,
    ) -> Result<Self, MtmdError> {
        let chunk_type = chunk.fit_to_micro_batch(micro_batch_tokens)?;

        Ok(Self {
            chunk,
            chunk_type,
            micro_batch_tokens,
        })
    }

    #[must_use]
    pub const fn chunk(&self) -> MtmdInputChunkView<'chunks, 'context> {
        self.chunk
    }

    #[must_use]
    pub const fn chunk_type(&self) -> MtmdInputChunkType {
        self.chunk_type
    }

    /// Evaluates the chunk like [`MtmdInputChunkView::eval`], without fitting it again: decodes
    /// at least as large as the micro batch it was fitted to fit it too.
    ///
    /// # Errors
    ///
    /// Returns any error of [`crate::mtmd_context::MtmdContext::micro_batch_tokens_in`],
    /// [`MtmdError::EvaluatedMicroBatchSmallerThanFitted`] when decodes of `n_batch` in
    /// `llama_ctx` are smaller than the micro batch the chunk was fitted to, or any error of
    /// [`MtmdInputChunkView::position_after`], all before evaluating anything, or an
    /// [`MtmdError`] when llama.cpp fails to encode or decode the chunk.
    pub fn eval_single(
        &self,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
        let evaluated_micro_batch_tokens = self
            .chunk
            .mtmd_context()
            .micro_batch_tokens_in(llama_ctx, n_batch, seq_id)?;

        if evaluated_micro_batch_tokens < self.micro_batch_tokens {
            return Err(MtmdError::EvaluatedMicroBatchSmallerThanFitted {
                evaluated_micro_batch_tokens,
                fitted_micro_batch_tokens: self.micro_batch_tokens,
            });
        }

        unsafe {
            self.chunk
                .eval_unchecked(llama_ctx, start_position, seq_id, n_batch, logits_last)
        }
    }
}
