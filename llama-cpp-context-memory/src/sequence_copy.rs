use llama_cpp_bindings_sys::bindings::llama_pos;
use llama_cpp_bindings_sys::bindings::llama_seq_id;

const fn reaches_cache_edge(position: llama_pos, cache_size: u32) -> bool {
    position <= 0 || position.cast_unsigned().saturating_add(1) >= cache_size
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceCopy {
    pub src: llama_seq_id,
    pub dest: llama_seq_id,
    pub p0: llama_pos,
    pub p1: llama_pos,
}

impl SequenceCopy {
    /// Whether the copy takes the whole source sequence: it starts at the first position and has
    /// no end bound.
    #[must_use]
    pub const fn is_whole_sequence(&self) -> bool {
        self.p0 <= 0 && self.p1 < 0
    }

    /// Whether neither bound leaves a cell of a `cache_size`-cell cache out of the copy, which
    /// llama.cpp requires of a copy between two KV streams.
    #[must_use]
    pub const fn spans_whole_cache(&self, cache_size: u32) -> bool {
        reaches_cache_edge(self.p0, cache_size) && reaches_cache_edge(self.p1, cache_size)
    }
}

#[cfg(test)]
mod tests {
    use super::SequenceCopy;

    const fn bounded(p0: i32, p1: i32) -> SequenceCopy {
        SequenceCopy {
            src: 0,
            dest: 1,
            p0,
            p1,
        }
    }

    #[test]
    fn only_a_copy_from_the_first_position_without_an_end_takes_the_whole_sequence() {
        assert_eq!(
            [
                bounded(-1, -1).is_whole_sequence(),
                bounded(0, -1).is_whole_sequence(),
                bounded(1, -1).is_whole_sequence(),
                bounded(0, 0).is_whole_sequence(),
            ],
            [true, true, false, false]
        );
    }

    #[test]
    fn a_copy_spans_the_whole_cache_when_each_bound_is_open_or_reaches_its_last_cell() {
        assert_eq!(
            [
                bounded(-1, -1).spans_whole_cache(256),
                bounded(0, 0).spans_whole_cache(256),
                bounded(255, -1).spans_whole_cache(256),
                bounded(0, 255).spans_whole_cache(256),
                bounded(254, -1).spans_whole_cache(256),
                bounded(0, 254).spans_whole_cache(256),
            ],
            [true, true, true, true, false, false]
        );
    }
}
