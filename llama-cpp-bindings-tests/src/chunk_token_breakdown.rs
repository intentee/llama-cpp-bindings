use anyhow::Context as _;
use anyhow::Result;
use llama_cpp_bindings::mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::mtmd_input_chunks::MtmdInputChunks;

pub struct ChunkTokenBreakdown {
    pub text: u64,
    pub image: u64,
    pub audio: u64,
}

impl ChunkTokenBreakdown {
    /// # Errors
    ///
    /// Fails when a chunk below the chunk count is missing or reports a type unknown to the
    /// bindings.
    pub fn from_chunks(chunks: &MtmdInputChunks) -> Result<Self> {
        let mut breakdown = Self {
            text: 0,
            image: 0,
            audio: 0,
        };

        for index in 0..chunks.len() {
            let chunk = chunks
                .get(index)
                .context("every index below the chunk count must hold a chunk")?;
            let n_tokens =
                u64::try_from(chunk.n_tokens()).context("a chunk's token count fits u64")?;

            let chunk_type_tokens = match chunk
                .chunk_type()
                .context("every chunk must have a known type")?
            {
                MtmdInputChunkType::Text => &mut breakdown.text,
                MtmdInputChunkType::Image => &mut breakdown.image,
                MtmdInputChunkType::Audio => &mut breakdown.audio,
            };

            *chunk_type_tokens = chunk_type_tokens
                .checked_add(n_tokens)
                .context("the tokens of one chunk type must fit u64")?;
        }

        Ok(breakdown)
    }
}
