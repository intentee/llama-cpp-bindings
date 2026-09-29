#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum JsonProbeOutcome {
    StillPossiblyValid,
    CompletedValid,
    Failed,
}
