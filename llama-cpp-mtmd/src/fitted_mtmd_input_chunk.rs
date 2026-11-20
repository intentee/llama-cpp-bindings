use std::num::NonZeroU32;

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
    /// type it classified, so the chunk can be ingested without classifying it again.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdInputChunkView::fit_to_micro_batch`].
    pub fn fit(
        chunk: MtmdInputChunkView<'chunks, 'context>,
        llama_ctx: &LlamaContext,
        batch_tokens: NonZeroU32,
    ) -> Result<Self, MtmdError> {
        let chunk_type = chunk.fit_to_micro_batch(llama_ctx, batch_tokens)?;

        Ok(Self {
            chunk,
            chunk_type,
            micro_batch_tokens: llama_ctx.limits().micro_batch_tokens(batch_tokens),
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

    /// Evaluates the chunk with [`MtmdInputChunkView::eval`] once decodes of `n_batch` in
    /// `llama_ctx` are at least as large as the micro batch the chunk was fitted to.
    /// [`MtmdInputChunkView::eval`] fits the chunk to `llama_ctx` and `n_batch` again, because
    /// whether the decodes of its pieces fit depends on both.
    ///
    /// # Errors
    ///
    /// Returns any error of [`crate::mtmd_context::MtmdContext::batch_tokens_in`], then
    /// [`MtmdError::EvaluatedMicroBatchSmallerThanFitted`] when decodes of `n_batch` in
    /// `llama_ctx` are smaller than the micro batch the chunk was fitted to, both before
    /// evaluating anything, then any error of [`MtmdInputChunkView::eval`].
    pub fn eval_single(
        &self,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
        let batch_tokens = self
            .chunk
            .mtmd_context()
            .batch_tokens_in(llama_ctx, n_batch, seq_id)?;
        let evaluated_micro_batch_tokens = llama_ctx.limits().micro_batch_tokens(batch_tokens);

        if evaluated_micro_batch_tokens < self.micro_batch_tokens {
            return Err(MtmdError::EvaluatedMicroBatchSmallerThanFitted {
                evaluated_micro_batch_tokens,
                fitted_micro_batch_tokens: self.micro_batch_tokens,
            });
        }

        self.chunk
            .eval(llama_ctx, start_position, seq_id, n_batch, logits_last)
    }
}
