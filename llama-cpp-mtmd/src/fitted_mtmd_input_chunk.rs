use std::marker::PhantomData;

use llama_cpp_context::llama_context::LlamaContext;

use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_eval_limits::MtmdEvalLimits;
use crate::mtmd_input_chunk::MtmdInputChunk;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::mtmd_input_chunks::MtmdInputChunks;

#[derive(Debug)]
pub struct FittedMtmdInputChunk<'fitted> {
    chunk: MtmdInputChunk,
    chunk_type: MtmdInputChunkType,
    fitted_chunks: PhantomData<&'fitted MtmdInputChunks>,
    micro_batch_tokens: u32,
}

impl FittedMtmdInputChunk<'_> {
    #[must_use]
    pub const fn new(
        chunk: MtmdInputChunk,
        chunk_type: MtmdInputChunkType,
        micro_batch_tokens: u32,
    ) -> Self {
        Self {
            chunk,
            chunk_type,
            fitted_chunks: PhantomData,
            micro_batch_tokens,
        }
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
    /// [`MtmdError::EvaluatedMicroBatchSmallerThanFitted`] when decodes of `n_batch` in
    /// `llama_ctx` are smaller than the micro batch the chunks were fitted to, or
    /// [`MtmdError::EvalFailed`] if the underlying encode or decode step fails.
    pub fn eval_single(
        &self,
        mtmd_ctx: &MtmdContext,
        llama_ctx: &mut LlamaContext,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<llama_cpp_bindings_sys::bindings::llama_pos, MtmdError> {
        let limits = MtmdEvalLimits::of(llama_ctx);
        let evaluated_micro_batch_tokens = limits
            .context
            .micro_batch_tokens(limits.batch_tokens(n_batch, seq_id)?);

        if evaluated_micro_batch_tokens < self.micro_batch_tokens {
            return Err(MtmdError::EvaluatedMicroBatchSmallerThanFitted {
                evaluated_micro_batch_tokens,
                fitted_micro_batch_tokens: self.micro_batch_tokens,
            });
        }

        self.chunk.eval(
            mtmd_ctx,
            llama_ctx,
            start_position,
            seq_id,
            n_batch,
            logits_last,
        )
    }
}
