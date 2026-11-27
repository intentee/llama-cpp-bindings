use serde::Deserialize;
use serde::Serialize;

use crate::decision_capabilities::DecisionCapabilities;
use crate::decisions_unsupported_reason::DecisionsUnsupportedReason;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DecisionsCapability {
    Supported(DecisionCapabilities),
    Unsupported(DecisionsUnsupportedReason),
}
