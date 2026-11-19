use std::borrow::Borrow;

use crate::fitted_mtmd_input_chunk::FittedMtmdInputChunk;
use crate::mtmd_context::MtmdContext;
use crate::mtmd_error::MtmdError;
use crate::mtmd_input_chunk_type::MtmdInputChunkType;
use crate::mtmd_input_chunks::MtmdInputChunks;

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
    ) -> Result<Self, MtmdError> {
        let chunk_types = chunks
            .borrow()
            .fit_to_micro_batch(mtmd_ctx, micro_batch_tokens)?;

        Ok(Self {
            chunk_types,
            chunks,
            micro_batch_tokens,
        })
    }

    /// Every chunk in order, each with the type it was fitted as.
    #[must_use]
    pub fn fitted_chunks(&self) -> Vec<FittedMtmdInputChunk<'_>> {
        self.chunks
            .borrow()
            .chunks()
            .into_iter()
            .zip(self.chunk_types.iter().copied())
            .map(|(chunk, chunk_type)| {
                FittedMtmdInputChunk::new(chunk, chunk_type, self.micro_batch_tokens)
            })
            .collect()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.chunk_types.is_empty()
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.chunk_types.len()
    }
}
