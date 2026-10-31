use crate::ingest_outcome::IngestOutcome;
use crate::streaming_json_probe::StreamingJsonProbe;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BareJsonToolCallProbeState {
    Idle,
    Holding {
        probe: StreamingJsonProbe,
        held: Vec<IngestOutcome>,
    },
}
