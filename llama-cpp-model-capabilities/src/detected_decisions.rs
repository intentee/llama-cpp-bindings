use llama_cpp_bindings_types::decisions_unsupported_reason::DecisionsUnsupportedReason;
use llama_cpp_decision::decision_model::DecisionModel;

#[derive(Clone)]
pub enum DetectedDecisions {
    Supported(DecisionModel),
    Unsupported(DecisionsUnsupportedReason),
}
