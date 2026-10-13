use std::borrow::Borrow;

use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunk_type::MtmdInputChunkType;
use super::mtmd_input_chunks::MtmdInputChunks;

#[derive(Debug)]
pub struct FittedMtmdInputChunks<TChunks>
where
    TChunks: Borrow<MtmdInputChunks>,
{
    chunk_types: Vec<MtmdInputChunkType>,
    chunks: TChunks,
    micro_batch_tokens: u32,
}

impl<TChunks> FittedMtmdInputChunks<TChunks>
where
    TChunks: Borrow<MtmdInputChunks>,
{
    /// Checks every chunk once with [`MtmdInputChunks::fit_to_micro_batch`], so the chunks can
    /// be evaluated and ingested without checking or classifying them again.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdInputChunks::fit_to_micro_batch`].
    pub fn fit(
        chunks: TChunks,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<Self, MtmdEvalError> {
        let chunk_types = chunks
            .borrow()
            .fit_to_micro_batch(mtmd_ctx, micro_batch_tokens)?;

        Ok(Self {
            chunk_types,
            chunks,
            micro_batch_tokens,
        })
    }

    #[must_use]
    pub fn get(&self, chunk_index: usize) -> Option<MtmdInputChunk> {
        self.chunks.borrow().get(chunk_index)
    }

    #[must_use]
    pub fn chunk_type(&self, chunk_index: usize) -> Option<MtmdInputChunkType> {
        self.chunk_types.get(chunk_index).copied()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.chunk_types.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.chunk_types.is_empty()
    }

    #[must_use]
    pub const fn micro_batch_tokens(&self) -> u32 {
        self.micro_batch_tokens
    }
}
