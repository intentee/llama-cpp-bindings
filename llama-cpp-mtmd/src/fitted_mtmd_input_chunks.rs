use llama_cpp_context::llama_context::LlamaContext;

use crate::evaluated_mtmd_input_chunk::EvaluatedMtmdInputChunk;
use crate::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_view::MtmdInputChunkView;

#[derive(Debug)]
pub struct FittedMtmdInputChunks<'chunks, 'context> {
    fitted: Vec<FittedMtmdInputChunk<'chunks, 'context>>,
}

impl<'chunks, 'context> FittedMtmdInputChunks<'chunks, 'context> {
    /// Checks every chunk once with [`FittedMtmdInputChunk::fit`], in chunk order, so the chunks
    /// can be evaluated and ingested without checking or classifying them again.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::NonCausalChunkExceedsMicroBatch`] for the first chunk decoded
    /// non-causally that has more tokens than `micro_batch_tokens`, or
    /// [`MtmdError::EvalUnknownChunkType`] when a chunk type is unknown.
    pub fn fit(
        chunks: Vec<MtmdInputChunkView<'chunks, 'context>>,
        micro_batch_tokens: u32,
    ) -> Result<Self, MtmdError> {
        let fitted = chunks
            .into_iter()
            .map(|chunk| FittedMtmdInputChunk::fit(chunk, micro_batch_tokens))
            .collect::<Result<Vec<FittedMtmdInputChunk<'chunks, 'context>>, MtmdError>>()?;

        Ok(Self { fitted })
    }

    /// Evaluates the chunks in order with [`FittedMtmdInputChunk::eval_single`], one chunk each
    /// time the returned iterator advances, from `start_position` on and with logits for the last
    /// token of the last chunk when `logits_last` asks for them. The iterator yields each
    /// evaluated chunk with the position after it, or the error evaluating it, at which callers
    /// stop, as `?` and [`Iterator::try_fold`] do.
    ///
    /// # Errors
    ///
    /// Returns [`MtmdError::ChunkPositionsOverflow`] for the first chunk whose positions
    /// overflow when the chunks are evaluated from `start_position`, before evaluating any of
    /// them. Each chunk yields any error of [`FittedMtmdInputChunk::eval_single`].
    pub fn eval_each<'fitted>(
        &'fitted self,
        llama_ctx: &'fitted mut LlamaContext<'_>,
        start_position: llama_cpp_bindings_sys::bindings::llama_pos,
        seq_id: llama_cpp_bindings_sys::bindings::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<
        impl Iterator<Item = Result<EvaluatedMtmdInputChunk<'fitted, 'chunks, 'context>, MtmdError>>,
        MtmdError,
    > {
        self.fitted
            .iter()
            .try_fold(start_position, |position, fitted_chunk| {
                fitted_chunk.chunk().position_after(position)
            })?;

        let last_chunk_index = self.fitted.len().checked_sub(1);

        Ok(self.fitted.iter().enumerate().scan(
            start_position,
            move |position, (index, fitted_chunk)| {
                Some(
                    fitted_chunk
                        .eval_single(
                            llama_ctx,
                            *position,
                            seq_id,
                            n_batch,
                            logits_last && Some(index) == last_chunk_index,
                        )
                        .map(|next_position| {
                            *position = next_position;

                            EvaluatedMtmdInputChunk {
                                fitted_chunk,
                                next_position,
                            }
                        }),
                )
            },
        ))
    }

    /// Every chunk in order, each with the type it was fitted as.
    #[must_use]
    pub const fn fitted_chunks(&self) -> &[FittedMtmdInputChunk<'chunks, 'context>] {
        self.fitted.as_slice()
    }
}
