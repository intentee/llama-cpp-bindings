use llama_cpp_bindings::ingest_outcome::IngestOutcome;
use llama_cpp_model::sampled_token_section::SampledTokenSection;

#[must_use]
pub fn visible_text_in(outcomes: &[IngestOutcome], section: SampledTokenSection) -> String {
    outcomes
        .iter()
        .filter(|IngestOutcome { sampled_token, .. }| sampled_token.section() == section)
        .map(|IngestOutcome { piece, .. }| piece.visible())
        .collect()
}
