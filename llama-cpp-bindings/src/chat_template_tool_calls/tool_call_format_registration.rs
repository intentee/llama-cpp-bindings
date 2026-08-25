use llama_cpp_bindings_types::ToolCallMarkers;

use crate::chat_template_tool_calls::gemma4_call_block::Gemma4CallBlockFormat;
use crate::chat_template_tool_calls::glm47_key_value_tags::Glm47KeyValueTagsFormat;
use crate::chat_template_tool_calls::mistral3_arrow_args::Mistral3ArrowArgsFormat;
use crate::chat_template_tool_calls::qwen_xml_tags::QwenXmlTagsFormat;
use crate::chat_template_tool_calls::qwen3_json_inside_tool_call::Qwen3JsonInsideToolCallFormat;

pub struct ToolCallFormatRegistration {
    pub detect: fn(&str) -> Option<ToolCallMarkers>,
    pub markers: fn() -> ToolCallMarkers,
}

impl ToolCallFormatRegistration {
    /// Every chat-template tool-call format this crate knows, ordered so that the more
    /// restrictive shapes are tried before the ones whose separators could greedily
    /// match them.
    pub const KNOWN: &'static [Self] = &[
        Self {
            detect: Qwen3JsonInsideToolCallFormat::detect,
            markers: Qwen3JsonInsideToolCallFormat::markers,
        },
        Self {
            detect: QwenXmlTagsFormat::detect,
            markers: QwenXmlTagsFormat::markers,
        },
        Self {
            detect: Glm47KeyValueTagsFormat::detect,
            markers: Glm47KeyValueTagsFormat::markers,
        },
        Self {
            detect: Mistral3ArrowArgsFormat::detect,
            markers: Mistral3ArrowArgsFormat::markers,
        },
        Self {
            detect: Gemma4CallBlockFormat::detect,
            markers: Gemma4CallBlockFormat::markers,
        },
    ];
}
