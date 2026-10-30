use llama_cpp_bindings::mtmd::MtmdInputChunkType;
use llama_cpp_bindings::mtmd::MtmdInputChunks;

pub struct ChunkTokenBreakdown {
    pub text: u64,
    pub image: u64,
    pub audio: u64,
}

impl ChunkTokenBreakdown {
    /// # Panics
    ///
    /// Panics when a chunk below the chunk count is missing or reports a type unknown to the
    /// bindings.
    #[must_use]
    pub fn from_chunks(chunks: &MtmdInputChunks) -> Self {
        let mut breakdown = Self {
            text: 0,
            image: 0,
            audio: 0,
        };

        for index in 0..chunks.len() {
            let chunk = chunks
                .get(index)
                .expect("every index below the chunk count must hold a chunk");
            let n_tokens = u64::try_from(chunk.n_tokens()).expect("a chunk's token count fits u64");

            match chunk
                .chunk_type()
                .expect("every chunk must have a known type")
            {
                MtmdInputChunkType::Text => breakdown.text += n_tokens,
                MtmdInputChunkType::Image => breakdown.image += n_tokens,
                MtmdInputChunkType::Audio => breakdown.audio += n_tokens,
            }
        }

        breakdown
    }
}
