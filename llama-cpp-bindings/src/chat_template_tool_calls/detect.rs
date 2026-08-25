use llama_cpp_bindings_types::ToolCallMarkers;

use crate::chat_template_tool_calls::tool_call_format_registration::ToolCallFormatRegistration;

#[must_use]
pub fn detect(template: &str) -> Option<ToolCallMarkers> {
    ToolCallFormatRegistration::KNOWN
        .iter()
        .find_map(|registration| (registration.detect)(template))
}

#[cfg(test)]
mod tests {
    use super::detect;
    use crate::chat_template_tool_calls::gemma4_call_block::Gemma4CallBlockFormat;
    use crate::chat_template_tool_calls::mistral3_arrow_args::Mistral3ArrowArgsFormat;
    use crate::chat_template_tool_calls::qwen_xml_tags::QwenXmlTagsFormat;

    #[test]
    fn detects_gemma4_call_block_format() {
        let template = "{{- '<|tool_call>call:' + function['name'] + '{' -}}";
        let markers = detect(template).expect("must dispatch to Gemma 4");

        assert_eq!(markers, Gemma4CallBlockFormat::markers());
    }

    #[test]
    fn detects_mistral3_arrow_args_format() {
        let template = "{{- name + '[ARGS]' + arguments }}";
        let markers = detect(template).expect("must dispatch to Mistral 3");

        assert_eq!(markers, Mistral3ArrowArgsFormat::markers());
    }

    #[test]
    fn detects_qwen_xml_tags_format() {
        let template = "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}";
        let markers = detect(template).expect("must dispatch to Qwen XML tags");

        assert_eq!(markers, QwenXmlTagsFormat::markers());
    }

    #[test]
    fn returns_none_when_no_known_format_matches() {
        assert!(detect("plain unrelated template").is_none());
    }
}
