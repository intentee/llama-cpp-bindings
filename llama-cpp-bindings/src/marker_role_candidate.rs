use crate::marker_role::MarkerRole;
use crate::token::LlamaToken;

#[derive(Clone, Debug, Eq, PartialEq)]
/// A tokenized marker string together with the single role it was detected as.
///
/// Candidates are merged by token sequence, so the same tokens detected under two
/// roles become one [`crate::streaming_marker::StreamingMarker`] carrying both.
pub struct MarkerRoleCandidate {
    pub tokens: Vec<LlamaToken>,
    pub role: MarkerRole,
}
