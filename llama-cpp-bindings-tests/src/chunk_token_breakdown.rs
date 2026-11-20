use anyhow::Context as _;
use anyhow::Result;

use llama_cpp_mtmd::mtmd_input_chunk_type::MtmdInputChunkType;
use llama_cpp_mtmd::mtmd_input_chunk_view::MtmdInputChunkView;
use llama_cpp_mtmd::mtmd_input_chunks::MtmdInputChunks;

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
    pub fn from_chunks(chunks: &MtmdInputChunks<'_>) -> Result<Self> {
        let empty = Self {
            text: 0,
            image: 0,
            audio: 0,
        };

        (0..chunks.len()).try_fold(empty, |breakdown, index| {
            chunks
                .get(index)
                .context("every index below the chunk count must hold a chunk")
                .and_then(|chunk| breakdown.with_chunk(chunk))
        })
    }

    fn with_chunk(mut self, chunk: MtmdInputChunkView<'_, '_>) -> Result<Self> {
        let n_tokens = chunk.n_tokens() as u64;

        chunk
            .chunk_type()
            .context("every chunk must have a known type")
            .map(|chunk_type| {
                let chunk_type_tokens = match chunk_type {
                    MtmdInputChunkType::Text => &mut self.text,
                    MtmdInputChunkType::Image => &mut self.image,
                    MtmdInputChunkType::Audio => &mut self.audio,
                };

                *chunk_type_tokens = chunk_type_tokens.saturating_add(n_tokens);

                self
            })
    }
}
