use std::borrow::Borrow;
use std::marker::PhantomData;

use crate::context::LlamaContext;

use super::eval_chunk::eval_chunk;
use super::fitted_mtmd_input_chunks::FittedMtmdInputChunks;
use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_eval_limits::MtmdEvalLimits;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunks::MtmdInputChunks;

#[derive(Debug)]
pub struct FittedMtmdInputChunk<'fitted> {
    chunk: MtmdInputChunk,
    chunk_type: MtmdInputChunkType,
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
        let chunk_type = fitted_chunks.chunk_type(chunk_index)?;

        fitted_chunks.get(chunk_index).map(|chunk| Self {
            chunk,
            chunk_type,
            fitted_chunks: PhantomData,
            micro_batch_tokens: fitted_chunks.micro_batch_tokens(),
        })
    }

    #[must_use]
    pub const fn chunk(&self) -> &MtmdInputChunk {
        &self.chunk
    }

    #[must_use]
    pub const fn chunk_type(&self) -> MtmdInputChunkType {
        self.chunk_type
    }

    /// # Errors
    ///
    /// Returns any error of [`MtmdEvalLimits::batch_tokens`],
    /// [`MtmdEvalError::EvaluatedMicroBatchSmallerThanFitted`] when decodes of `n_batch` in
    /// `llama_ctx` are smaller than the micro batch the chunks were fitted to, or
    /// [`MtmdEvalError::EvalFailed`] if the underlying encode or decode step fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::llama_pos,
        seq_id: llama_cpp_bindings_sys::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::llama_pos, MtmdEvalError> {
        let limits = MtmdEvalLimits::of(llama_ctx);
        let evaluated_micro_batch_tokens = limits
            .context
            .micro_batch_tokens(limits.batch_tokens(n_batch, seq_id)?);

        if evaluated_micro_batch_tokens < self.micro_batch_tokens {
            return Err(MtmdEvalError::EvaluatedMicroBatchSmallerThanFitted {
                evaluated_micro_batch_tokens,
                fitted_micro_batch_tokens: self.micro_batch_tokens,
            });
        }

        eval_chunk(
            &self.chunk,
            mtmd_ctx,
            llama_ctx,
            start_position,
            seq_id,
            n_batch,
            logits_last,
        )
    }
}
