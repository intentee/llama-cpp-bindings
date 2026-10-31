use crate::bare_json_tool_call_probe::BareJsonToolCallProbe;
use crate::ingest_outcome::IngestOutcome;

pub enum ContentProbe {
    Disabled,
    BareJson(BareJsonToolCallProbe),
}

impl ContentProbe {
    pub fn pass(&mut self, outcome: IngestOutcome, released: &mut Vec<IngestOutcome>) {
        match self {
            Self::Disabled => released.push(outcome),
            Self::BareJson(probe) => probe.pass(outcome, released),
        }
    }

    pub fn release(&mut self, released: &mut Vec<IngestOutcome>) {
        match self {
            Self::Disabled => {}
            Self::BareJson(probe) => probe.release(released),
        }
    }
}
