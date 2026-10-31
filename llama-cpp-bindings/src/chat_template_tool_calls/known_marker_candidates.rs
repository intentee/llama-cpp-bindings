use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

use crate::chat_template_tool_calls::gemma4_call_block::Gemma4CallBlockFormat;
use crate::chat_template_tool_calls::glm47_key_value_tags::Glm47KeyValueTagsFormat;
use crate::chat_template_tool_calls::mistral3_arrow_args::Mistral3ArrowArgsFormat;
use crate::chat_template_tool_calls::qwen_xml_tags::QwenXmlTagsFormat;
use crate::chat_template_tool_calls::qwen3_json_inside_tool_call::Qwen3JsonInsideToolCallFormat;

/// Every chat-template tool-call format this crate knows, ordered so that the more restrictive
/// shapes are tried before the ones whose separators could greedily match them.
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
