use llama_cpp_bindings_sys::bindings::llama_rs_memory_kind;

use crate::context_memory_error::ContextMemoryError;
use crate::sequence_copy::SequenceCopy;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLayout {
    pub kind: llama_rs_memory_kind,
    pub kv_unified: bool,
    pub cache_size: u32,
}

impl MemoryLayout {
    /// # Errors
    ///
    /// Refuses the copies llama.cpp aborts the process on: a partial copy in a memory that copies
    /// only whole sequences, a partial copy between sequences that keep separate KV streams, and
    /// any copy in a memory whose kind these bindings do not recognize.
    pub const fn check_copy(&self, copy: &SequenceCopy) -> Result<(), ContextMemoryError> {
        match self.kind {
            llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSV4
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID_IDX => {
                self.check_whole_sequence_copy(copy)
            }
            llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_ISWA
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA_ISWA
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_MSA
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID
            | llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID_ISWA => {
                self.check_kv_stream_copy(copy)
            }
            llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_RECURRENT => Ok(()),
            llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_UNRECOGNIZED => {
                Err(ContextMemoryError::SeqCpUnrecognizedMemoryKind)
            }
        }
    }

    const fn check_kv_stream_copy(&self, copy: &SequenceCopy) -> Result<(), ContextMemoryError> {
        let SequenceCopy { src, dest, p0, p1 } = *copy;

        if self.kv_unified || src == dest || copy.spans_whole_cache(self.cache_size) {
            return Ok(());
        }

        Err(ContextMemoryError::SeqCpPartialCopyAcrossKvStreams {
            src,
            dest,
            p0,
            p1,
            cache_size: self.cache_size,
        })
    }

    const fn check_whole_sequence_copy(
        &self,
        copy: &SequenceCopy,
    ) -> Result<(), ContextMemoryError> {
        if copy.is_whole_sequence() {
            return Ok(());
        }

        let SequenceCopy { p0, p1, .. } = *copy;

        Err(ContextMemoryError::SeqCpPartialCopyOfWholeSequenceMemory {
            memory_kind: self.kind,
            p0,
            p1,
        })
    }
}

#[cfg(test)]
mod tests {
    use llama_cpp_bindings_sys::bindings::llama_rs_memory_kind;

    use super::MemoryLayout;
    use crate::context_memory_error::ContextMemoryError;
    use crate::sequence_copy::SequenceCopy;

    const CACHE_SIZE: u32 = 256;

    const WHOLE_SEQUENCE_ONLY_KINDS: [llama_rs_memory_kind; 2] = [
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSV4,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID_IDX,
    ];

    const KV_STREAM_KINDS: [llama_rs_memory_kind; 7] = [
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_ISWA,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_DSA_ISWA,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE_MSA,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID,
        llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_HYBRID_ISWA,
    ];

    const PARTIAL_COPY_TO_ANOTHER_SEQUENCE: SequenceCopy = SequenceCopy {
        src: 0,
        dest: 1,
        p0: 0,
        p1: 1,
    };

    const WHOLE_COPY_TO_ANOTHER_SEQUENCE: SequenceCopy = SequenceCopy {
        src: 0,
        dest: 1,
        p0: 0,
        p1: -1,
    };

    const fn separate_streams(kind: llama_rs_memory_kind) -> MemoryLayout {
        MemoryLayout {
            kind,
            kv_unified: false,
            cache_size: CACHE_SIZE,
        }
    }

    #[test]
    fn memory_that_copies_only_whole_sequences_refuses_a_partial_copy() {
        assert_eq!(
            WHOLE_SEQUENCE_ONLY_KINDS
                .map(|kind| separate_streams(kind).check_copy(&PARTIAL_COPY_TO_ANOTHER_SEQUENCE)),
            WHOLE_SEQUENCE_ONLY_KINDS.map(|memory_kind| Err(
                ContextMemoryError::SeqCpPartialCopyOfWholeSequenceMemory {
                    memory_kind,
                    p0: 0,
                    p1: 1,
                }
            ))
        );
    }

    #[test]
    fn memory_that_copies_only_whole_sequences_accepts_a_whole_copy() {
        assert_eq!(
            WHOLE_SEQUENCE_ONLY_KINDS
                .map(|kind| separate_streams(kind).check_copy(&WHOLE_COPY_TO_ANOTHER_SEQUENCE)),
            [Ok(()), Ok(())]
        );
    }

    #[test]
    fn kv_caches_refuse_a_partial_copy_between_separate_streams() {
        assert_eq!(
            KV_STREAM_KINDS
                .map(|kind| separate_streams(kind).check_copy(&PARTIAL_COPY_TO_ANOTHER_SEQUENCE)),
            KV_STREAM_KINDS.map(
                |_kind| Err(ContextMemoryError::SeqCpPartialCopyAcrossKvStreams {
                    src: 0,
                    dest: 1,
                    p0: 0,
                    p1: 1,
                    cache_size: CACHE_SIZE,
                })
            )
        );
    }

    #[test]
    fn kv_caches_accept_a_partial_copy_that_stays_within_one_stream() {
        let unified = MemoryLayout {
            kv_unified: true,
            ..separate_streams(llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE)
        };
        let same_sequence = SequenceCopy {
            dest: 0,
            ..PARTIAL_COPY_TO_ANOTHER_SEQUENCE
        };

        assert_eq!(
            [
                unified.check_copy(&PARTIAL_COPY_TO_ANOTHER_SEQUENCE),
                separate_streams(llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_KV_CACHE)
                    .check_copy(&same_sequence),
            ],
            [Ok(()), Ok(())]
        );
    }

    #[test]
    fn kv_caches_accept_a_copy_spanning_the_whole_cache_between_separate_streams() {
        assert_eq!(
            KV_STREAM_KINDS
                .map(|kind| separate_streams(kind).check_copy(&WHOLE_COPY_TO_ANOTHER_SEQUENCE)),
            KV_STREAM_KINDS.map(|_kind| Ok(()))
        );
    }

    #[test]
    fn recurrent_memory_accepts_a_partial_copy_between_separate_streams() {
        assert_eq!(
            separate_streams(llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_RECURRENT)
                .check_copy(&PARTIAL_COPY_TO_ANOTHER_SEQUENCE),
            Ok(())
        );
    }

    #[test]
    fn memory_of_an_unrecognized_kind_refuses_even_a_whole_copy() {
        assert_eq!(
            separate_streams(llama_rs_memory_kind::LLAMA_RS_MEMORY_KIND_UNRECOGNIZED)
                .check_copy(&WHOLE_COPY_TO_ANOTHER_SEQUENCE),
            Err(ContextMemoryError::SeqCpUnrecognizedMemoryKind)
        );
    }
}
