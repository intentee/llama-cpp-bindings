use std::borrow::Borrow;

use super::mtmd_context::MtmdContext;
use super::mtmd_eval_error::MtmdEvalError;
use super::mtmd_input_chunk::MtmdInputChunk;
use super::mtmd_input_chunks::MtmdInputChunks;

#[derive(Debug)]
pub struct FittedMtmdInputChunks<TChunks>
where
    TChunks: Borrow<MtmdInputChunks>,
{
    chunks: TChunks,
    micro_batch_tokens: u32,
}

impl<TChunks> FittedMtmdInputChunks<TChunks>
where
    TChunks: Borrow<MtmdInputChunks>,
{
    /// Checks every chunk once with [`MtmdInputChunks::fit_to_micro_batch`], so the chunks can
    /// be evaluated without checking them again.
    ///
    /// # Errors
    ///
    /// Returns any error of [`MtmdInputChunks::fit_to_micro_batch`].
    pub fn fit(
        chunks: TChunks,
        mtmd_ctx: &MtmdContext,
        micro_batch_tokens: u32,
    ) -> Result<Self, MtmdEvalError> {
        chunks
            .borrow()
            .fit_to_micro_batch(mtmd_ctx, micro_batch_tokens)?;

        Ok(Self {
            chunks,
            micro_batch_tokens,
        })
    }

    #[must_use]
    pub fn get(&self, chunk_index: usize) -> Option<MtmdInputChunk> {
        self.chunks.borrow().get(chunk_index)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.chunks.borrow().len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.borrow().is_empty()
    }

    #[must_use]
    pub const fn micro_batch_tokens(&self) -> u32 {
        self.micro_batch_tokens
    }
}
