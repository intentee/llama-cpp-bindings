use llama_cpp_bindings_types::ToolCallMarkers;

use crate::chat_template_tool_calls::gemma4_call_block::Gemma4CallBlockFormat;
use crate::chat_template_tool_calls::glm47_key_value_tags::Glm47KeyValueTagsFormat;
use crate::chat_template_tool_calls::mistral3_arrow_args::Mistral3ArrowArgsFormat;
use crate::chat_template_tool_calls::qwen_xml_tags::QwenXmlTagsFormat;
use crate::chat_template_tool_calls::qwen3_json_inside_tool_call::Qwen3JsonInsideToolCallFormat;

#[must_use]
pub fn known_marker_candidates() -> Vec<ToolCallMarkers> {
    vec![
        Qwen3JsonInsideToolCallFormat::markers(),
        QwenXmlTagsFormat::markers(),
        Glm47KeyValueTagsFormat::markers(),
        Mistral3ArrowArgsFormat::markers(),
        Gemma4CallBlockFormat::markers(),
    ]
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use llama_cpp_bindings_types::ToolCallArgsShape;

    use super::known_marker_candidates;

    #[test]
    fn known_marker_candidates_returns_one_per_registered_shape() {
        let candidates = known_marker_candidates();
        assert_eq!(candidates.len(), 5);

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
            5,
            "duplicate shape discriminants in known_marker_candidates: {shape_discriminants:?}"
        );
    }
}
