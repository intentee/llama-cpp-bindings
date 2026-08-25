use crate::ingest_outcome::IngestOutcome;
use crate::token::LlamaToken;

#[derive(Clone, Debug)]
pub struct ClassifiedSample {
    pub token: LlamaToken,
    pub outcomes: Vec<IngestOutcome>,
}
