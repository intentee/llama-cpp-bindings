use llama_cpp_bindings_types::key_value_xml_tags_shape::KeyValueXmlTagsShape;
use llama_cpp_bindings_types::tool_call_args_shape::ToolCallArgsShape;
use llama_cpp_bindings_types::tool_call_markers::ToolCallMarkers;

pub struct Glm47KeyValueTagsFormat;

impl Glm47KeyValueTagsFormat {
    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::KeyValueXmlTags(KeyValueXmlTagsShape {
                key_open: "<arg_key>".to_owned(),
                key_close: "</arg_key>".to_owned(),
                value_open: "<arg_value>".to_owned(),
                value_close: "</arg_value>".to_owned(),
            }),
        }
    }
}
