use llama_cpp_bindings_types::ToolCallArgsShape;
use llama_cpp_bindings_types::ToolCallMarkers;
use llama_cpp_bindings_types::XmlTagsShape;

pub struct QwenXmlTagsFormat;

impl QwenXmlTagsFormat {
    #[must_use]
    pub fn markers() -> ToolCallMarkers {
        ToolCallMarkers {
            open: "<tool_call>".to_owned(),
            close: "</tool_call>".to_owned(),
            args_shape: ToolCallArgsShape::XmlTags(XmlTagsShape {
                function_open_prefix: "<function=".to_owned(),
                function_close: "</function>".to_owned(),
                parameter_open_prefix: "<parameter=".to_owned(),
                parameter_close: "</parameter>".to_owned(),
            }),
        }
    }
}
