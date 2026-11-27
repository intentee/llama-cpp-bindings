#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SequenceOutput {
    ExceedsCapacity {
        n_seq_max: u32,
        output_capacity: u32,
    },
    FitsCapacity,
}
