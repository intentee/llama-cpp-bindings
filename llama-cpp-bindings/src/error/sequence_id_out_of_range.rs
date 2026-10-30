#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("sequence id {seq_id} names no sequence of a context with {n_seq_max} sequences")]
pub struct SequenceIdOutOfRange {
    pub seq_id: i32,
    pub n_seq_max: u32,
}
