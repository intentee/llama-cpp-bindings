use serde::Deserialize;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum ToolCallingCapability {
    OneCallPerTurn,
    ParallelCalls,
    Unsupported,
}
