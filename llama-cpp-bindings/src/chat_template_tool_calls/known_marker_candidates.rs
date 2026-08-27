use llama_cpp_bindings_types::ToolCallMarkers;

use crate::chat_template_tool_calls::tool_call_format_registration::ToolCallFormatRegistration;

#[must_use]
pub fn known_marker_candidates() -> Vec<ToolCallMarkers> {
    ToolCallFormatRegistration::KNOWN
        .iter()
        .map(|registration| (registration.markers)())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::known_marker_candidates;
    use crate::chat_template_tool_calls::tool_call_format_registration::ToolCallFormatRegistration;

    #[test]
    fn known_marker_candidates_returns_one_per_registered_shape() {
        let candidates = known_marker_candidates();

        assert_eq!(candidates.len(), ToolCallFormatRegistration::KNOWN.len());

        let shape_discriminants: HashSet<&'static str> = candidates
            .iter()
            .map(|markers| match &markers.args_shape {
                ToolCallArgsShape::BracketedJson(_) => "BracketedJson",
                ToolCallArgsShape::JsonObject(_) => "JsonObject",
                ToolCallArgsShape::KeyValueXmlTags(_) => "KeyValueXmlTags",
                ToolCallArgsShape::PairedQuote(_) => "PairedQuote",
                ToolCallArgsShape::XmlTags(_) => "XmlTags",
            })
            .collect();

        assert_eq!(
            shape_discriminants.len(),
            candidates.len(),
            "duplicate shape discriminants in known_marker_candidates: {shape_discriminants:?}"
        );
    }
}
