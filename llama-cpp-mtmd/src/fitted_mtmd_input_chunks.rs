use crate::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunks::MtmdInputChunks;

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
        chunks: &'chunks MtmdInputChunks<'context>,
        micro_batch_tokens: u32,
    ) -> Result<Self, MtmdError> {
        let fitted = chunks
            .chunks()
            .into_iter()
            .map(|chunk| FittedMtmdInputChunk::fit(chunk, micro_batch_tokens))
            .collect::<Result<Vec<FittedMtmdInputChunk<'chunks, 'context>>, MtmdError>>()?;

        Ok(Self { fitted })
    }

    /// Every chunk in order, each with the type it was fitted as.
    #[must_use]
    pub const fn fitted_chunks(&self) -> &[FittedMtmdInputChunk<'chunks, 'context>] {
        self.fitted.as_slice()
    }
}
