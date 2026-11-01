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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use llama_cpp_bindings_types::parsed_tool_call::ParsedToolCall;
    use llama_cpp_bindings_types::parsed_tool_call_id::ParsedToolCallId;
    use llama_cpp_bindings_types::tool_call_arguments::ToolCallArguments;
    use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

    use super::known_marker_candidates;
    use crate::chat_template_tool_calls::gemma4_call_block::Gemma4CallBlockFormat;
    use crate::chat_template_tool_calls::glm47_key_value_tags::Glm47KeyValueTagsFormat;
    use crate::chat_template_tool_calls::mistral3_arrow_args::Mistral3ArrowArgsFormat;
    use crate::chat_template_tool_calls::qwen_xml_tags::QwenXmlTagsFormat;
    use crate::tool_call_format::tool_call_format_outcome::ToolCallFormatOutcome;
    use crate::tool_call_format::try_parse;

    fn first_candidate_parsing(body: &str) -> Option<ToolCallMarkers> {
        known_marker_candidates().into_iter().find(|candidate| {
            matches!(try_parse(body, candidate), ToolCallFormatOutcome::Parsed(_))
        })
    }

    fn weather_in_paris() -> ToolCallFormatOutcome {
        ToolCallFormatOutcome::Parsed(vec![ParsedToolCall::new(
            ParsedToolCallId::NotProvidedByModel,
            "get_weather".to_owned(),
            ToolCallArguments::ValidJson(json!({"location": "Paris"})),
        )])
    }

    #[test]
    fn no_known_candidate_parses_plain_content() {
        let plain_content = "Sorry, I cannot help with that request.";

        for candidate in known_marker_candidates() {
            assert_eq!(
                try_parse(plain_content, &candidate),
                ToolCallFormatOutcome::NoMatch,
                "expected NoMatch for plain content under candidate {candidate:?}"
            );
        }
    }

    #[test]
    fn qwen_xml_input_resolves_to_the_qwen_xml_tags_format() {
        let qwen_input = "<tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>\n\
            Paris\n\
            </parameter>\n\
            </function>\n\
            </tool_call>";
        let resolved = first_candidate_parsing(qwen_input);

        assert_eq!(
            resolved,
            Some(QwenXmlTagsFormat::markers()),
            "duck-type ordering must resolve Qwen XML via the XmlTags shape (most restrictive \
             shape that requires `<function=`)"
        );
        assert_eq!(
            resolved.map(|candidate| try_parse(qwen_input, &candidate)),
            Some(weather_in_paris())
        );
    }

    #[test]
    fn glm_input_resolves_to_the_glm47_key_value_tags_format() {
        let glm_input = "<tool_call>get_weather\
            <arg_key>location</arg_key>\
            <arg_value>Paris</arg_value>\
            </tool_call>";
        let resolved = first_candidate_parsing(glm_input);

        assert_eq!(resolved, Some(Glm47KeyValueTagsFormat::markers()));
        assert_eq!(
            resolved.map(|candidate| try_parse(glm_input, &candidate)),
            Some(weather_in_paris())
        );
    }

    #[test]
    fn mistral_input_resolves_to_the_mistral3_bracketed_json_format() {
        let mistral_input = r#"[TOOL_CALLS]get_weather[ARGS]{"location":"Paris"}"#;
        let resolved = first_candidate_parsing(mistral_input);

        assert_eq!(
            resolved,
            Some(Mistral3ArrowArgsFormat::markers()),
            "the candidate ordering must try BracketedJson before PairedQuote because \
             PairedQuote's `{{` separator could greedily match Mistral's JSON args"
        );
        assert_eq!(
            resolved.map(|candidate| try_parse(mistral_input, &candidate)),
            Some(weather_in_paris())
        );
    }

    #[test]
    fn gemma_input_resolves_to_the_gemma4_paired_quote_format() {
        let gemma_input = "<|tool_call>call:get_weather{location:<|\"|>Paris<|\"|>}";
        let resolved = first_candidate_parsing(gemma_input);

        assert_eq!(resolved, Some(Gemma4CallBlockFormat::markers()));
        assert_eq!(
            resolved.map(|candidate| try_parse(gemma_input, &candidate)),
            Some(weather_in_paris())
        );
    }
}
